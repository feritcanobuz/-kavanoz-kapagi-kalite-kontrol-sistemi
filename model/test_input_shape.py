from keras.models import load_model

model = load_model("C:/Users/ferit/quality_control_project/data/split/model_final_cnn.keras")
print("Model input shape:", model.input_shape)
