from cv2 import cv2
import os
from shapely.geometry import Polygon, Point
from imgaug import augmenters as iaa
from imgaug.augmentables.polys import *
import json
import base64
import time
import concurrent.futures


def hamTong(a,chiso, path_input, path_output):
    print('a : ', a)
    print('path_input : ', path_input)

    a = a.strip(".json")

    def save_coco_json(instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)  # indent=2

    def save_enhenced_pic(enhanced_pic, save_pic_path):
        cv2.imwrite(save_pic_path, enhanced_pic)

    def make_polys(json_file):
        with open(json_file, "r") as js:
            json_data = json.load(js)
        polys = []
        for shape in json_data['shapes']:
            assert shape['shape_type'] == "polygon"
            polys.append(Polygon(shape['points'], label=shape['label']))
        img_shape = (json_data['imageHeight'], json_data['imageWidth'], 3)
        polys_oi = PolygonsOnImage(polys, shape=img_shape)
        return polys_oi, json_data

    # Xác định toán tử nâng cao
    my_augment = iaa.Sequential([

        iaa.WithBrightnessChannels(iaa.Add((-50, 50)))  # dang dung


        #---------zoom-------------
        # iaa.Affine(scale=2)  # 2

        # iaa.AffineCv2(scale=2.0)
        # iaa.PerspectiveTransform(scale=(0.01, 0.15), keep_size=False)

        #--------------lat trai , phai----------------
        # iaa.Fliplr(0.5)

        # --------------lat tren , duoi----------------
        # iaa.Rot90([1, 3])

    ])

    quokka_img = cv2.imread(f'{path_input}/{a}.jpg')
    polys_oi, jsondata = make_polys(f'{path_input}/{a}.json')

    # Đa giác ban đầu
    oringin_rings = polys_oi.items
    # print('oringin_rings : ', polys_oi.items)
    quokka_img = cv2.cvtColor(quokka_img, cv2.COLOR_BGR2RGB)
    # Nâng cao
    augmented = my_augment(image=quokka_img, polygons=polys_oi)
    # Chỉ để vẽ
    overlay_quokka = polys_oi.draw_on_image(quokka_img)

    augmented_list = my_augment(image=quokka_img, polygons=polys_oi)

    enhenced_img = augmented_list[0]
    enhenced_polys = augmented_list[1]

    overlaid_list = enhenced_polys.draw_on_image(enhenced_img)
    # ia.imshow(overlaid_list)

    enhenced_rings = enhenced_polys.items

    try:
        new_polys = enhenced_polys.clip_out_of_image()
        new_ring = new_polys.items  # clip ：Đa giác mới từ đa giác nâng cao
        overlaid_list2 = new_polys.draw_on_image(enhenced_img)
        # ia.imshow(overlaid_list2)

        ####################Tạo lại tệp json được đánh dấu bởi labelme ###############################

        # Để xem các điểm trong tệp json ban đầu có phải là danh sách kép hay không,
        # Nhận thấy rằng nó thực sự là một danh sách hai tầng
        t = jsondata['shapes']

        # Xây dựng một danh sách mới để lưu trữ từ điển
        shapes_list = []
        dict1 = ["label", "points", "group_id", "shape_type", "flags"]

        #
        # Nhận bộ sưu tập điểm trong mỗi đa giác sau khi cường hóa
        points = [list(ring_idx.coords) for ring_idx in new_ring]
        # Khả năng hiểu danh sách lồng nhau, mức độ hiểu danh sách lồng nhau được đánh giá dựa trên những điều sau đây cho
        points_list = [[d2.tolist() for d2 in d1] for d1 in points]

        # Lấy nhãn danh mục tương ứng với mỗi đa giác
        label = [ring_idx.label for ring_idx in new_ring]

        # Vòng lặp để tạo giá trị tương ứng với mỗi khóa hình đa giác
        dict2 = []
        dict_all = []

        # Vòng lặp để tạo các hình dạng tương ứng với mỗi đa giác
        for b in range(len(points_list)):
            dict2.append(label[b])  # "label"
            dict2.append(points_list[b])  # "points"
            dict2.append(None)  # Nonetype
            dict2.append("polygon")
            dict2.append({})  # Mệnh đề trống
            dict_all.append(dict2)
            dict2 = []

        for dict_child in dict_all:
            zipped = zip(dict1, dict_child)
            dict_final = dict(zipped)
            shapes_list.append(dict_final)

        ## Sửa đổi các hình dạng trong tệp json ban đầu, phần còn lại không thay đổi
        # print(jsondata)
        jsondata["shapes"] = shapes_list
        # Thực hiện mã hóa và mã hóa base64 trên hình ảnh nâng cao
        enhenced_img = enhenced_img[:, :, ::-1]
        enhenced_img_bytes = np.array(cv2.imencode('.jpg', enhenced_img)[1].tobytes())
        base64_data = base64.b64encode(enhenced_img_bytes)
        jiema = base64_data.decode('utf-8')
        hso = 2
        jsondata["imagePath"] = f'{a}_Fliplr.jpg'
        jsondata["imageData"] = jiema
        # Lưu tệp json đã sửa đổi thành

        save_coco_json(jsondata, f"{path_output}/{a}_BrightnessChannels.json")

        # ia.imshow(enhenced_img)
        save_enhenced_pic(enhenced_img, f'{path_output}/{a}_BrightnessChannels.jpg')
        print(f'{chiso} : xong anh {a}')
    except:
        pass


if __name__ == '__main__':

    path_input = 'D:\Code_Python_All\Data_Augmentation_Label\Test/anh_goc' # đường dẫn folder chứa file Json cần convert
    path_output = 'D:\Code_Python_All\Data_Augmentation_Label\Test/anh_biendoi'
    json_files = [pos_json for pos_json in os.listdir(path_input) if pos_json.endswith('.json')]

    with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as executor:
        start_time = time.perf_counter()
        list(executor.map(hamTong, json_files, [i for i in range(len(json_files))], [path_input for i in range(len(json_files))], [path_output for i in range(len(json_files))] ))
        finish_time = time.perf_counter()
    print(f"Program finished in {finish_time - start_time} seconds")
