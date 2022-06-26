import torch
import torchaudio

from data import AudioLight
from network import LightModel
from train import Root_dir, ann_file, Sample_rate, num_sample


class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    # load back the model
    cnn = LightModel()
    
    cnn.load_from_checkpoint(checkpoint_path='/content/audio-class.ckpt')

    # load urban sound dataset dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=Sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    batch_Size = 128
    usd = AudioLight(ann_file,Root_dir,Sample_rate,num_sample,batch_Size)

    # get a sample from the urban sound dataset for inference
    input, target = usd.data[10][0], usd.data[10][1] # [batch size, num_channels, fr, time]
    input.unsqueeze_(0)

    # make an inference
    predicted, expected = predict(cnn, input, target,
                                  class_mapping)
    print(f"Predicted: '{predicted}', expected: '{expected}'")