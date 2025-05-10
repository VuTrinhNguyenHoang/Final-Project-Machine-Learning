from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from .explainer import overlay_gradCAM, deprocess_image, Lime
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

W, H = 224, 224
DATA_DIR = 'data/cat_dog'
UPLOAD_DIR = 'website/static/uploads'
EXPLAIN_DIR = 'website/static/explained'

def explain(cnn, gradCAM, guidedBP, image_path, retrieved_path, im, y_pred):
    #################### Test Image #####################
    image = cv2.imread(os.path.join(UPLOAD_DIR, image_path))
    upsample_size = (image.shape[1], image.shape[0])

    # Overlayed GradCAM
    cam3 = gradCAM.compute_heatmap(im, y_pred, upsample_size)
    img_gradCAM = overlay_gradCAM(image, cam3)
    img_gradCAM = cv2.cvtColor(img_gradCAM, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 6))
    plt.imshow(img_gradCAM)
    plt.title('GradCAM Test Image', fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(EXPLAIN_DIR, 'gradCAM_{}'.format(image_path)), dpi=150)
    plt.close()

    # Guided GradCAM
    gb = guidedBP.guided_backprop(im, upsample_size)
    guided_gradcam = deprocess_image(gb * cam3)
    guided_gradcam = cv2.cvtColor(guided_gradcam, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 6))
    plt.imshow(guided_gradcam)
    plt.title('Guided GradCAM Test Image', fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(EXPLAIN_DIR, 'guidedGrad_{}'.format(image_path)), dpi=150)
    plt.close()
    ################## Retrieved Image ##################

    # Original
    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Test Image', fontsize=20)
    plt.axis('off')

    image = cv2.imread(os.path.join(DATA_DIR, retrieved_path))
    upsample_size = (image.shape[1], image.shape[0])

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Retrieved Image', fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(EXPLAIN_DIR, 'retrieved_img.jpg'), dpi=150)
    plt.close()

    # Overlayed GradCAM
    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.imshow(img_gradCAM)
    plt.title('GradCAM Test Image', fontsize=20)
    plt.axis('off')

    img = img_to_array(load_img(os.path.join(DATA_DIR, retrieved_path), target_size=(W, H)))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    preds = cnn.predict(img)
    idx = preds.argmax()

    cam3 = gradCAM.compute_heatmap(img, idx, upsample_size)
    img_gradCAM = overlay_gradCAM(image, cam3)
    img_gradCAM = cv2.cvtColor(img_gradCAM, cv2.COLOR_BGR2RGB)

    plt.subplot(122)
    plt.imshow(img_gradCAM)
    plt.title('GradCAM Retrieved Image', fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(EXPLAIN_DIR, 'retrieved_gradCAM.jpg'), dpi=150)
    plt.close()

    # Guided GradCAM
    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.imshow(guided_gradcam)
    plt.title('Guided GradCAM Test Image', fontsize=20)
    plt.axis('off')

    gb = guidedBP.guided_backprop(img, upsample_size)
    guided_gradcam = deprocess_image(gb * cam3)
    guided_gradcam = cv2.cvtColor(guided_gradcam, cv2.COLOR_BGR2RGB)
    plt.subplot(122)
    plt.imshow(guided_gradcam)
    plt.title('Guided GradCAM Retrieved Image', fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(EXPLAIN_DIR, 'retrieved_guidedGrad.jpg'), dpi=150)
    plt.close()

    # # LIME
    im = cv2.imread(os.path.join(UPLOAD_DIR, image_path))
    
    lime = Lime(cnn)
    explaination = lime.explain(im, 2)

    vis, mask = explaination.get_image_and_mask(
        y_pred,
        positive_only=True,
        num_features=5,
        hide_rest=True
    )
    vis = (vis / 255.0).astype(np.float32)
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.imshow(mark_boundaries(vis, mask))
    plt.title('Explain by LIME', fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(EXPLAIN_DIR, 'Lime_{}'.format(image_path)), dpi=150)
    plt.close()

    return [
        os.path.join('uploads', image_path),
        os.path.join('explained', 'gradCAM_{}'.format(image_path)),
        os.path.join('explained', 'guidedGrad_{}'.format(image_path)),
        os.path.join('explained', 'Lime_{}'.format(image_path)),
        os.path.join('explained', 'retrieved_img.jpg'),
        os.path.join('explained', 'retrieved_gradCAM.jpg'),
        os.path.join('explained', 'retrieved_guidedGrad.jpg')
    ]

