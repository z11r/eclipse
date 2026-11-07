# Architecture credits

Eclipse's code is written against its own `Frame` and `Detection` contract. No code,
assets or interfaces from these projects are in it. The ideas below are general ones,
listed here because a review of Eclipse pointed to these projects as examples of them.
We know them from that review's summary, not from reading their source.

| Idea in Eclipse | Where | Example the review pointed to |
| --- | --- | --- |
| Separate blocked and not-yet-clean states, with repeated checks before clearing | `compositor/gate.py`, `text_detect/covers.py` | [PII Guard / stream-guard, PrivacyGate](https://github.com/pc-style/stream-guard/blob/6bf8ce89ae5eb0e7a5a46c9c30eca0717601b834/Sources/PIIGuardCore/PrivacyGate.swift) |
| Following motion between OCR reads, with confidence checks | `text_detect/detection.py` | [OpenScrub](https://github.com/austinmabry/OpenScrub/blob/4f863f394ea0bccaf2eb7d35bd347802a6244ee9/openscrub.py#L1064) |
| Handing native pixel buffers straight to Vision | `platform.py` | [Moblin](https://github.com/eerimoq/moblin/blob/52d4f1250daae9c96d1be7da554de43113e77a5e/Moblin/Media/HaishinKit/Media/Video/VideoUnit.swift#L845) |
| Bounded pending OCR work | `compositor/delay.py` | [SensitiveGuard](https://github.com/CodeBhaukaal/obs-sensitive-guard/blob/8fdb0be5fcef9f0d5e7248d6858b604dae4c4f4d/src/capture/frame-sampler.cpp) |

The scroll check's split between proposing a motion and clearing pixels, the output gate
and policy generations, and held-content scoring follow behaviour described in that
review and in our own earlier prototype. They were written fresh for this codebase.
