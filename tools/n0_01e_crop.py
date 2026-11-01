from PIL import Image
import sys
out=sys.argv[1]; fs=sys.argv[2:]
ims=[Image.open(f).crop((0,170,1280,520)) for f in fs]
s=Image.new('RGB',(1280,350*len(ims)))
for i,im in enumerate(ims): s.paste(im,(0,i*350))
s.save(out)
