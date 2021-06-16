import tensorflow as tf
import glob
import copy
import cv2
import numpy as np
import json

model_path = 'C:/Users/kosakae256/Documents/Kosakae-Deployment/CollectNumberPlates/model/model2/saved_model'
DEFAULT_FUNCTION_KEY = 'serving_default'
loaded_model = tf.saved_model.load(model_path)
inference_func = loaded_model.signatures[DEFAULT_FUNCTION_KEY]

# 推論用関数(Function for inference)
def run_inference_single_image(image, inference_func):
    tensor = tf.convert_to_tensor(image)
    output = inference_func(tensor)

    output['num_detections'] = int(output['num_detections'][0])
    output['detection_classes'] = output['detection_classes'][0].numpy()
    output['detection_boxes'] = output['detection_boxes'][0].numpy()
    output['detection_scores'] = output['detection_scores'][0].numpy()
    return output

def detection_number(readpath,detectpath,formpath,jsonpath,filename):
    image = cv2.imread(readpath)
    debug_image = copy.deepcopy(image)

    image_width, image_height = image.shape[1], image.shape[0]
    image = image[:, :, [2, 1, 0]]  # BGR2RGB
    image_np_expanded = np.expand_dims(image, axis=0)

    output = run_inference_single_image(image_np_expanded, inference_func)

    c = 0
    num_detections = output['num_detections']

    detectdict = {
            "bbox" : [],
            "filename" : filename,
            "class" : [],
            }

    class_names = ["number","car"]
    for i in range(num_detections):
        score = output['detection_scores'][i]
        bbox = output['detection_boxes'][i]
        labelclass = output['detection_classes'][i]

        #この数値以上の信頼度なら通る
        if score < 0.3:
            continue
        print(labelclass)
        x1, y1 = int(bbox[1] * image_width), int(bbox[0] * image_height)
        x2, y2 = int(bbox[3] * image_width), int(bbox[2] * image_height)
        detectdict["bbox"].append([x1,y1,x2,y2])
        detectdict["class"].append(class_names[int(labelclass)-1])

        # 推論結果描画(Inference result drawing)
        cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(debug_image, str('{:.2f}'.format(score)) + class_names[int(labelclass)-1], (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
        c+=1

    #どこにも判定がなければ
    if c!=0:
        cv2.imwrite(f"{detectpath}/detect{filename}",debug_image)
        image = image[:, :, [2, 1, 0]]  # RGB2BGR
        cv2.imwrite(f"{formpath}/form{filename}",image)
        with open(f'{jsonpath}/{filename}.json', 'w') as f: # 第二引数：writableオプションを指定
            json.dump(detectdict, f)

if __name__ == "__main__":
    detectpath = "C:/Users/kosakae256/Documents/Kosakae-Deployment/CollectNumberPlates/detectimgs"
    formpath = "C:/Users/kosakae256/Documents/Kosakae-Deployment/CollectNumberPlates/formimgs"
    jsonpath = "C:/Users/kosakae256/Documents/Kosakae-Deployment/CollectNumberPlates/detectjsons"
    temp_data_dir = 'C:/Users/kosakae256/Documents/Kosakae-Deployment/CollectNumberPlates/tempimgs'
    tempfile_list = sorted(glob.glob(temp_data_dir + '/*.jpg'))
    for filecount, fpath in enumerate(tempfile_list):
        fpath = fpath.replace("\\","/")
        filename = fpath.split("/")[-1]
        print(filename)
        detection_number(fpath,detectpath,formpath,jsonpath,filename)
