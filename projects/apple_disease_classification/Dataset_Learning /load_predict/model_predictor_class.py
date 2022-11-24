def classify(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)

    img_batch = np.expand_dims(img_array, axis=0)

    img_preprocessed = preprocess_input(img_batch)

    model = tf.keras.models.load_model(
        'saved_models/tl_mobileNetV2/model/tl_22112022_12h10_1.h5', custom_objects=None, compile=True, options=None)

    classifier = tf.keras.Sequential([
        hub.KerasLayer(
            "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=img_size+(3,))
    ])
    prediction = model.predict(img_preprocessed)

    print(decode_predictions(prediction, top=3)[0])


classify("../data/80apples/naamloze map/AnyConv.com__download (15).jpg")
