from PIL import Image,ImageDraw

anno_box_path = r"D:/AIstudyCode/data/CelebA/Anno/list_bbox_celeba.txt"
label_dir = "D:/AIstudyCode/data/CelebA/labels"
img_dir = "D:/AIstudyCode/data/CelebA/Img/img_celeba.7z/img_celeba.7z/img_celeba"
count = 0
epoch = 1
box_file = open(anno_box_path,"r")

i = 0


for line in box_file:
    if i < 2:
        i += 1
        continue
    i += 1
    print(line)

    imgname = line[0:6]
    #print(imgname)

    img_strs = line.split()
    x1, y1, w, h = int(img_strs[1]), int(img_strs[2]), int(img_strs[3]), int(img_strs[4])
    x2, y2 = x1+w, y1+h

    img = Image.open(f"{img_dir}/{img_strs[0]}")
    img_w, img_h = img.size

    # ****************************
    dw = 1. / (int(img_w))
    dh = 1. / (int(img_h))
    x = ((x1 + x2) / 2.0 - 1)*dw
    y = ((y1 + y2) / 2.0 - 1)*dh
    w = (x2 - x1)*dw
    h = (y2 - y1)*dh
    # x = x * dw
    # w = w * dw
    # y = y * dh
    # h = h * dh
    # ****************************
    label_txt = open(f"{label_dir}/{imgname}.txt", "w")

    label_txt.write(f"0 {x} {y} {w} {h}\n")
    label_txt.flush()
    label_txt.close()

    if i == 1562:
        exit()