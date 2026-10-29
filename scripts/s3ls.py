import sys, urllib.request, urllib.parse, re
B="https://vesuvius-challenge-open-data.s3.amazonaws.com/"
def ls(prefix, delim="/"):
    out=[];tok=None
    while True:
        q={"list-type":"2","prefix":prefix,"delimiter":delim}
        if tok: q["continuation-token"]=tok
        x=urllib.request.urlopen(B+"?"+urllib.parse.urlencode(q),timeout=60).read().decode()
        out+=re.findall(r"<Prefix>([^<]*)</Prefix>",x)[1:] if delim else []
        out+=re.findall(r"<Key>([^<]*)</Key>",x)
        m=re.search(r"<NextContinuationToken>([^<]*)<",x)
        if not m: return out
        tok=m.group(1)
if __name__=="__main__":
    for p in ls(sys.argv[1], "/" if len(sys.argv)<3 else sys.argv[2]): print(p)
