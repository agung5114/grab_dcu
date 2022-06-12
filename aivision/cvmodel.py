import tensorflow
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.mobilenet import decode_predictions

from PIL import Image
sys.modules['Image'] = Image

def getPrediction(data,model):
    img = Image.open(data)
    newsize = (224, 224)
    image = img.resize(newsize)
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = yhat[0]
    prob = []
    for i in range(len(label)):
        # prob.append(i)
        prob.append(np.round(label[i]*100,2))
    data = {'Food': food, 'Prob': prob}
    # return data
    dfhasil = pd.DataFrame.from_dict(data)

    dfhasil['Probability'] = dfhasil.apply(lambda x: f"{x['Prob']}%", axis=1)
    top3 = dfhasil.nlargest(3, 'Prob')
    # top = dict(zip(food, prob))
    # top3 = dict(sorted(top.items(), key=operator.itemgetter(1), reverse=True)[:3])
    return top3
