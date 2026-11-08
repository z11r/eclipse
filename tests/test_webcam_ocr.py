from concurrent.futures import ThreadPoolExecutor
import sys
import time

import numpy as np
import pytest

from eclipse import main as app
from eclipse.capture.dual import pip_layout
from eclipse.text_detect.apple_process import AppleProcessReader
from eclipse.text_detect.read_ahead import ReadError
from eclipse.types import Frame


class Pool:
    def __init__(self, fail=False):
        self.stats = {}
        self.fail = fail
        self.closed = 0
        self.active = 0
        self.peak = 0

    def submit(self, image, urgent=False):
        assert urgent
        self.active += 1
        self.peak = max(self.peak, self.active)
        value = int(image[0,0,0])
        pool = self
        class Job:
            def result(self):
                try:
                    time.sleep(.002)
                    if pool.fail:
                        raise RuntimeError('FAKE_PRIVATE_VALUE')
                    return [([[0,0],[10,0],[10,10],[0,10]], str(value), .9)]
                finally:
                    pool.active -= 1
        return Job()

    def close(self): self.closed += 1


def test_shared_reader_serializes_crops_and_keeps_results_with_their_images():
    pool = Pool()
    reader = AppleProcessReader(pool=pool)
    images = [np.full((20,30,3), value, np.uint8) for value in range(8)]
    with ThreadPoolExecutor(max_workers=4) as threads:
        results = list(threads.map(reader, images))
    assert [words[0][1] for words, _ in results] == [str(i) for i in range(8)]
    assert pool.peak == 1
    reader.close()
    reader.close()
    assert pool.closed == 1
    with pytest.raises(ReadError): reader(images[0])


def test_failed_worker_is_not_a_clean_empty_read():
    reader = AppleProcessReader(pool=Pool(fail=True))
    with pytest.raises(ReadError) as caught:
        reader(np.zeros((20,30,3),np.uint8))
    assert 'FAKE_PRIVATE_VALUE' not in str(caught.value)
    reader.close()


def test_retina_scan_is_sized_for_final_inset_and_can_restore_capture_sizing():
    screen, camera = (3384,6016,3), (720,1280,3)
    at, shown, scan = pip_layout(screen,camera,.25,output_size=(1920,1080))
    old_at, old_shown, old_scan = pip_layout(screen,camera,.25,output_size=None)
    assert at == old_at and shown == old_shown
    assert scan == (960,540) and old_scan == (1280,720)
    assert scan[0]*scan[1] / (old_scan[0]*old_scan[1]) == pytest.approx(.5625)


def test_scan_sizing_accounts_for_letterboxing_and_never_upscales_camera():
    _, shown, scan = pip_layout((4000,6016,3),(720,1280,3),.25,output_size=(1920,1080))
    assert scan[0] == 813  # output height determines final fit, not output width
    _, _, small = pip_layout((3384,6016,3),(240,320,3),.25,output_size=(1920,1080))
    assert small == (320,240)


def test_apple_camera_selection_rejected_before_opening_camera_off_macos(monkeypatch):
    monkeypatch.setattr(app,'IS_MAC',False)
    monkeypatch.setattr(app,'AppleProcessReader',lambda: pytest.fail('must not spawn worker'))
    with pytest.raises(SystemExit) as caught:
        app.main(['--webcam','0','--webcam-ocr','apple'])
    assert caught.value.code == 2


def test_webcam_ocr_option_needs_camera_input(monkeypatch):
    monkeypatch.setattr(app,'IS_MAC',True)
    with pytest.raises(SystemExit):
        app.main(['--webcam-ocr','apple'])


class NativeReader:
    def __init__(self): self.closed = 0; self.calls = 0; self.stats = {}
    def __call__(self,image):
        self.calls += 1
        return [([[20,20],[200,20],[200,40],[20,40]],'demo@example.com',.99)], None
    def close(self): self.closed += 1


class Feed:
    def __init__(self, source='webcam', shape=(240,320,3)):
        self.source=source;self.shape=shape;self.index=0
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def read(self):
        result=Frame(np.full(self.shape,150,np.uint8),time.monotonic(),self.source,self.index)
        self.index+=1
        return result


class Sink:
    def __init__(self,*args,**kwargs): self.images=[]
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def send(self,image): self.images.append(image)


def test_main_routes_webcam_reads_to_native_process_and_releases_it(monkeypatch,tmp_path):
    reader=NativeReader();sink=Sink()
    monkeypatch.setattr(app,'IS_MAC',True)
    monkeypatch.setattr(app,'AppleProcessReader',lambda:reader)
    monkeypatch.setattr(app,'WebcamCapture',lambda index:Feed())
    monkeypatch.setattr(app,'VirtualCam',lambda **kwargs:sink)
    assert app.main(['--webcam','0','--webcam-ocr','apple','--max-frames','4',
                     '--delay','.05','--fps','60','--log',str(tmp_path/'run.log')]) == 0
    assert reader.calls > 0 and reader.closed == 1
    assert sink.images and all(not image[20:40,20:200].any() for image in sink.images)


def test_native_reader_closed_if_later_startup_fails(monkeypatch):
    reader=NativeReader()
    monkeypatch.setattr(app,'IS_MAC',True)
    monkeypatch.setattr(app,'AppleProcessReader',lambda:reader)
    def fail(**kwargs): raise RuntimeError('detector startup failed')
    monkeypatch.setattr(app,'TextDetector',fail)
    with pytest.raises(RuntimeError):
        app.main(['--webcam','0','--webcam-ocr','apple','--max-frames','1'])
    assert reader.closed == 1


def test_combined_views_share_one_native_camera_reader(monkeypatch,tmp_path):
    reader=NativeReader()
    monkeypatch.setattr(app,'IS_MAC',True)
    monkeypatch.setattr(app,'AppleProcessReader',lambda:reader)
    monkeypatch.setattr(app,'WebcamCapture',lambda index:Feed())
    monkeypatch.setattr(app,'ScreenCapture',lambda **kwargs:Feed('screen'))
    monkeypatch.setattr(app,'screen_capture_allowed',lambda:True)
    monkeypatch.setattr(app,'screen_capture_kit_available',lambda:False)
    monkeypatch.setattr(app,'VirtualCam',lambda **kwargs:Sink())
    used=[]
    real_detector=app.TextDetector
    def make(**kwargs):
        used.append(kwargs.get('reader'))
        if 'reader' not in kwargs: kwargs['reader']=lambda image:([],None)
        return real_detector(**kwargs)
    monkeypatch.setattr(app,'TextDetector',make)
    assert app.main(['--with-webcam','0','--webcam-ocr','apple','--max-frames','3',
                     '--delay','.05','--fps','60','--log',str(tmp_path/'run.log')]) == 0
    assert used == [None,reader,reader]
    assert reader.closed == 1


@pytest.mark.skipif(sys.platform!='darwin',reason='native Apple Vision needs macOS')
def test_real_camera_process_recognizes_synthetic_email():
    import cv2
    image=np.full((720,1280,3),240,np.uint8)
    cv2.putText(image,'demo@example.com',(100,300),cv2.FONT_HERSHEY_SIMPLEX,1.6,(20,20,20),3)
    reader=AppleProcessReader()
    try:
        words,_=reader(image)
        assert any('example.com' in word[1] for word in words)
    finally:
        reader.close()


def test_default_layout_preserves_existing_scan_detail():
    _,_,scan=pip_layout((3384,6016,3),(720,1280,3),.25)
    assert scan == (1280,720)


@pytest.mark.skipif(sys.platform!='darwin',reason='native Apple Vision needs macOS')
def test_native_webcam_rules_on_synthetic_sensitive_text():
    import cv2
    from eclipse.text_detect import TextDetector
    image=np.full((720,1280,3),240,np.uint8)
    rows=['demo@example.com','Phone: +1 (780) 555-0123','OTP: 482913',
          'PASSWORD=fake-pass-123','4111 1111 1111 1111']
    for row,text in enumerate(rows):
        cv2.putText(image,text,(60,120+row*100),cv2.FONT_HERSHEY_SIMPLEX,1.,(20,20,20),2)
    reader=AppleProcessReader()
    try:
        found=TextDetector(reader=reader,noise_floor=8)(Frame(image,0,'webcam',0))
        assert {'email','phone','two_fa','api_key','card'} <= {d.category for d in found}
    finally:
        reader.close()


def test_help_lists_camera_options_without_starting_resources(monkeypatch,capsys):
    monkeypatch.setattr(app,'AppleProcessReader',lambda:pytest.fail('help must not start OCR'))
    with pytest.raises(SystemExit) as caught:
        app.main(['--help'])
    assert caught.value.code == 0
    help_text=capsys.readouterr().out
    assert '--webcam-ocr' in help_text and '--webcam-scan' in help_text


def test_invalid_lookahead_rejected_before_native_reader_startup(monkeypatch):
    monkeypatch.setattr(app,'IS_MAC',True)
    monkeypatch.setattr(app,'AppleProcessReader',lambda:pytest.fail('invalid args must not start OCR'))
    with pytest.raises(SystemExit) as caught:
        app.main(['--webcam','0','--webcam-ocr','apple','--delay','.5','--look-ahead','1'])
    assert caught.value.code == 2
