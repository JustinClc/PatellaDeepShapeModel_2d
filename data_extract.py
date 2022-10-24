import os

img_list = []
mask_list = []
id_list = []
for path in glob.glob('./MOST_lateral_seg_masks/*/*/*/*/*/*'):
    original_img = os.path.join(path, 'input0.png')
    mask = os.path.join(path, 'output2.png')
    img_id = path.split('\\')[2]
    img_list.append(original_img)
    mask_list.append(mask)
    id_list.append(img_id)