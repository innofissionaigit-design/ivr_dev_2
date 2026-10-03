"""Hear every voice the TTS server offers, so the agent's voice can be chosen by ear (KCD-511).

The agent's voice is the TTS_SPEAKER setting; the current one is a female voice. If the clinic wants a
mature (40 plus) female voice, the way to find one is to LISTEN: this tool asks the running TTS server
which speakers it has (GET /speakers) and synthesises the same line in each, one wav file per
speaker and language, so a person can play them and pick. Nothing here can judge how old a voice
sounds; a person has to.

    python tools/audition_voices.py --url http://localhost:8002 --out voices/

Then set TTS_SPEAKER=<name> in deploy/env.sh and restart the TTS service.
"""

import argparse
import os
import sys

import httpx

LINES = {
    "bn": "নমস্কার। আমি সোনোস্ক্যান বাণী। বলুন, কীভাবে সাহায্য করতে পারি?",
    "hi": "नमस्कार। मैं सोनोस्कैन वाणी हूँ। बताइए, मैं आपकी क्या मदद कर सकती हूँ?",
    "en": "Hello. I am Sonoscan Vaani. How can I help you?",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8002")
    ap.add_argument("--out", default="voices")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    with httpx.Client(base_url=args.url, timeout=60.0) as c:
        info = c.get("/speakers").json()
        print("in use:", info["default"])
        for lang, names in info["speakers"].items():
            for name in names or [info["default"]]:
                r = c.post("/synthesize", json={"lang": lang, "text": LINES[lang], "speaker": name})
                if r.status_code != 200:
                    print(f"  {lang}/{name}: skipped ({r.status_code})")
                    continue
                path = os.path.join(args.out, f"{lang}_{name}.wav")
                with open(path, "wb") as f:
                    f.write(r.content)
                print("  wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
