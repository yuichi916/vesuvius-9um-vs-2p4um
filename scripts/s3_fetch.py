"""Mirror a prefix of the public vesuvius-challenge-open-data S3 bucket.

usage: python -I s3_fetch.py PREFIX DEST_DIR [workers]
  PREFIX: key prefix ending with '/', e.g. PHerc0139/segments/.../x.zarr/
"""
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HOST = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_keys(prefix):
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(f"{HOST}/?{urllib.parse.urlencode(q)}", timeout=60) as r:
            root = ET.fromstring(r.read())
        for c in root.findall("s:Contents", NS):
            keys.append((c.find("s:Key", NS).text, int(c.find("s:Size", NS).text)))
        if root.find("s:IsTruncated", NS).text != "true":
            return keys
        token = root.find("s:NextContinuationToken", NS).text


def fetch(key, size, prefix, dest):
    out = dest / key[len(prefix):]
    if out.exists() and out.stat().st_size == size:
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{HOST}/{urllib.parse.quote(key)}", timeout=60) as r:
                data = r.read()
            if len(data) != size:
                raise IOError(f"size {len(data)} != {size}")
            tmp = out.with_name(out.name + ".part")
            tmp.write_bytes(data)
            tmp.replace(out)
            return size
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                print(f"FAIL {key}: {e}", flush=True)
                return -1
            time.sleep(2 ** attempt)


def main():
    prefix, dest = sys.argv[1], Path(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    keys = list_keys(prefix)
    print(f"{len(keys)} keys, {sum(s for _, s in keys)/1e9:.2f} GB", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda ks: fetch(ks[0], ks[1], prefix, dest), keys))
    fails = sum(1 for r in res if r < 0)
    got = sum(r for r in res if r > 0)
    print(f"DONE {got/1e6:.0f} MB new in {time.time()-t0:.0f}s, fails={fails}", flush=True)


if __name__ == "__main__":
    main()
