import json
import os
from yt_extractor import get_video_info, get_audio_url
from api import save_transcript


def save_video_sentiments(url):
    video_info = get_video_info(url)
    url = get_audio_url(video_info)
    if url:
        title = video_info['title']
        title = title.strip().replace(" ", "_")
        title = "data/" + title
        save_transcript(url, title, sentiment_analysis=True)

if __name__ == "__main__":
    # save_video_sentiments("https://youtu.be/MD0k7aWS5Fc?si=VW1dm4s8A4sWU9yN")

    data_folder = 'data'
    filename = "data_format.json"

    file_path = os.path.join(data_folder, filename)

    with open(file_path, "r") as f:
        data = json.load(f)
    
    positives = []
    negatives = []
    neutrals = []
    for result in data:
        text = result["text"]
        if result["sentiment"] == "POSITIVE":
            positives.append(text)
        elif result["sentiment"] == "NEGATIVE":
            negatives.append(text)
        else:
            neutrals.append(text)
        
    n_pos = len(positives)
    n_neg  = len(negatives)
    n_neut = len(neutrals)

    print("Num positives:", n_pos)
    print("Num negatives:", n_neg)
    print("Num neutrals:", n_neut)

    if n_pos + n_neg > 0:
        r = n_pos / (n_pos + n_neg)
        print(f"Positive ratio: {r:.3f}")
    else:
        print("No positive or negative sentiments found.")