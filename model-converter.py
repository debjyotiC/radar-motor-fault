from edgemodelkit import ModelPlayGround

# Initialize the ModelPlayGround with the path to your .keras model
playground = ModelPlayGround()

playground.load_model(model_path="saved-model/best-radar-motor-fault.keras")

# playground.model_summary()

playground.model_stats()

playground.model_converter(quantization_type="float16")