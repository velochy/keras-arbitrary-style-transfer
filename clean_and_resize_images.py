from os import walk, path, remove
from PIL import Image
import sys

max_dim = 600

count, bad = 0, 0
for root,_,files in walk(path.abspath(sys.argv[1])):
    for file in files:
        filename = path.join(root,file) 
        if filename.endswith('.jpg'):
            count += 1
            if count%100==0: print(count)

            try:
                img = Image.open(filename) # open the image file
                img.verify() # verify that it is, in fact an image

                width, height = img.size
                has_alpha = (len(img.getbands())==4)
                
                if width>max_dim or height>max_dim or has_alpha:
                    #print("Shrinking %s" % filename)
                    img.close() # Reopen    
                    img = Image.open(filename)

                    # Remove alpha channel (as JPG does not support it)
                    if has_alpha:
                        nimg = Image.new("RGB", img.size, (255, 255, 255))
                        nimg.paste(img, mask=img.split()[3]) # 3 is the alpha channel
                        img.close()
                        img = nimg

                    img.thumbnail((max_dim,max_dim))
                    img.save(filename, "JPEG")
                    
                img.close()

            except Exception as e:
                print('Bad file:', filename) # print out the names of corrupt files
                remove(filename)
                bad += 1

print("Total: %i Bad: %i" % (count, bad))