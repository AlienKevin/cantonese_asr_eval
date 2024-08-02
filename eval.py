import json
from evaluate import load
import os
import matplotlib.pyplot as plt

cer_metric = load("cer")

def calculate_cer(expected, transcription):
    """
    Calculate the Character Error Rate (CER) between two strings.
    """
    return cer_metric.compute(predictions=[transcription], references=[expected])

def convert_to_simplified(text):
    """
    Convert traditional Chinese text to simplified Chinese using opencc.
    """
    from opencc import OpenCC
    converter = OpenCC('t2s')
    return converter.convert(text)

def remove_emotion_and_event_tokens(text):
    """
    Remove emotion and event tokens from text.
    """
    emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
    event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷",}
    return ''.join(char for char in text if char not in emo_set and char not in event_set)

def remove_punctuations(text):
    """
    Remove punctuations from text.
    """
    import regex
    return regex.sub(r'\p{P}+', '', text)

def remove_whitespaces(text):
    """
    Remove whitespaces from text.
    """
    import regex
    return regex.sub(r'\p{Separator}+', '', text)

def eval(transcriptions_path):
    # Read the transcriptions.json file
    with open(transcriptions_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Calculate CER for each entry after converting expected sentences to simplified Chinese
    total_cer = 0
    count = 0

    for entry in data:
        expected_simplified = remove_whitespaces(remove_punctuations(convert_to_simplified(entry['expected'])))
        generated = remove_whitespaces(remove_punctuations(convert_to_simplified(remove_emotion_and_event_tokens(entry['transcription']))))
        cer = calculate_cer(expected_simplified, generated)
        # print(f'Expected: {expected_simplified} | Transcription: {generated} | CER: {cer}')
        total_cer += cer
        count += 1

    average_cer = total_cer / count if count > 0 else 0
    return average_cer

def eval_all_results():
    from tqdm import tqdm

    results_dir = 'results'
    evaluation_results = {}

    # Calculate the total number of JSON files to process
    total_files = sum(len(files) for _, _, files in os.walk(results_dir) if any(file.endswith('.json') for file in files))

    with tqdm(total=total_files, desc="Evaluating results") as pbar:
        for model_name in os.listdir(results_dir):
            model_dir = os.path.join(results_dir, model_name)
            if os.path.isdir(model_dir):
                for dataset_file in os.listdir(model_dir):
                    if dataset_file.endswith('.json'):
                        dataset_name = dataset_file.replace('.json', '')
                        cer = eval(os.path.join(model_dir, dataset_file))
                        if dataset_name not in evaluation_results:
                            evaluation_results[dataset_name] = {}
                        evaluation_results[dataset_name][model_name] = cer * 100
                        pbar.update(1)

    return evaluation_results

def plot_evaluation_results(evaluation_results, dataset_tasks):
    dataset_labels = []
    cer_values = {}

    for dataset in dataset_tasks.keys():
        task = dataset_tasks.get(dataset, "Unknown Task")
        dataset_label = task
        dataset_labels.append(dataset_label)
        
        models = evaluation_results.get(dataset, {})
        for model_name, cer in models.items():
            if model_name not in cer_values:
                cer_values[model_name] = []
            cer_values[model_name].append(cer)

    x = range(len(dataset_labels))
    width = 0.35
    model_names = ['sensevoice_small', 'whisper_large_v2_cantonese_scrya', 'whisper_small_cantonese']
    colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999']  # Define a list of color-blind friendly colors

    plt.figure(figsize=(14, 8))
    group_spacing = 0.5  # Add spacing between groups
    for i, model_name in enumerate(model_names):
        plt.bar([p * (1 + group_spacing) - width * (len(model_names) / 2) + width * i + width / 2 for p in x],
                cer_values[model_name],
                width,
                label=model_name,
                color=colors[i % len(colors)])

    plt.xlabel('Domains', fontsize=18)
    plt.ylabel('CER (%)', fontsize=18)
    plt.title('Evaluation of Open Source Cantonese ASR Models in Diverse Domains', fontsize=22)
    plt.xticks([p * (1 + group_spacing) for p in x], dataset_labels, rotation=0, ha='center', fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluation_results = eval_all_results()
    for dataset, models in evaluation_results.items():
        print(f'Evaluating {dataset}')
        for model, cer in models.items():
            print(f'{model} CER: {cer:.2f}%')

    dataset_tasks = {
        "common_voice_17_0": "Mixed",
        "guangzhou_daily_use": "Daily Use",
        "guangzhou_cabin": "Commands",
        "mixed_cantonese_and_english": "Yue & Eng",
        "zoengjyutgaai_saamgwokjinji": "Storytelling",
        "wordshk_hiujin": "Synthetic",
    }

    plot_evaluation_results(evaluation_results, dataset_tasks)
