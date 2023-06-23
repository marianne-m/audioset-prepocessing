import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Optional, List
import logging
import tqdm
import pandas as pd
from pyannote.audio import Pipeline

from scripts.datasets import NoiseSegment, AudioSet


MUSIC_IDS = range(137,283)
FORBIDDEN_LABELS = {
  "Speech",
  "Male speech, man speaking",
  "Female speech, woman speaking",
  "Child speech, kid speaking",
  "Conversation",
  "Narration, monologue",
  "Babbling",
  "Speech synthesizer",
  "Shout",
  "Bellow",
  "Whoop",
  "Yell",
  "Battle cry",
  "Children shouting",
  "Screaming",
  "Whispering",
  "Laughter",
  "Baby laughter",
  "Giggle",
  "Snicker",
  "Belly laugh",
  "Chuckle, chortle",
  "Crying, sobbing",
  "Baby cry, infant cry",
  "Whimper",
  "Wail, moan",
  "Sigh",
  "Singing",
  "Choir",
  "Yodeling",
  "Chant",
  "Mantra",
  "Male singing",
  "Female singing",
  "Child singing",
  "Synthetic singing",
  "Rapping",
  "Humming",
  "Groan",
  "Grunt",
  "Whistling",
  "Breathing",
  "Wheeze",
  "Snoring",
  "Gasp",
  "Pant",
  "Snort",
  "Cough",
  "Throat clearing",
  "Sneeze",
  "Sniff",
  "Gargling",
  "Burping, eructation",
  "Hiccup",
  "Cheering",
  "Chatter",
  "Crowd",
  "Hubbub, speech noise, speech babble",
  "Children playing",
  "A capella",
  "Beatboxing",
  "Pop music",
  "Hip hop music",
  "Rock music",
  "Heavy metal",
  "Punk rock",
  "Grunge",
  "Progressive rock",
  "Rock and roll",
  "Psychedelic rock",
  "Rhythm and blues",
  "Soul music",
  "Reggae",
  "Country",
  "Opera",
  "Techno",
  "Trance music",
  "Music for children",
  "Vocal music",
  "Christian music",
  "Music of Bollywood",
  "Gospel music",
  "Song"
}


def vad(pipeline: Pipeline, audiofile: Path):
    output = pipeline(audiofile)
    return len(output) > 0


def parser(argv):
    parser = argparse.ArgumentParser(description='Process AudioSet to create a "clean" subset.')

    parser.add_argument('metadata', type=Path,
                        help="Path to the metadata")
    parser.add_argument('label_path', type=Path,
                        help="Path to mid - label names csv")
    parser.add_argument('data_path', type=Path,
                        help="Path to the audiofiles")
    parser.add_argument('--debug', default=False, action="store_true",
                        help="Use debug mode with only 100 files")

    return parser.parse_args(argv)


def main(argv):
    args = parser(argv)

    dataset = AudioSet(args.metadata, label_path=args.label_path)
    pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                    use_auth_token="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")

    # check if multiple labels
    metadata = dataset.metadata[dataset.metadata.labels.map(len) == 1]

    if args.debug:
        metadata = metadata[:100]

    # check if label is not forbidden
    metadata = metadata[metadata.labels.map(lambda x: x[0] not in FORBIDDEN_LABELS)]

    # downsampling music
    df_music = pd.read_csv(args.label_path)
    dict_music = df_music["display_name"].to_dict()
    dict_music = {dict_music[x] for x in MUSIC_IDS}

    metadata["is_music"] = metadata.labels.map(lambda x: x[0] in dict_music)
    music = metadata[metadata["is_music"]]
    # music = music.sample(25200)
    music = music.sample(10)
    not_music = metadata[~metadata["is_music"]]
    dataset.metadata = pd.concat([music, not_music]) \
        .sort_values("file_id") \
        .reset_index(drop=True) \
        .drop(columns='is_music')

    # check if file exsts
    dataset.drop_files_not_found(args.data_path)

    # check if vad
    filepath = dataset.filepath(args.data_path)
    dataset.metadata = dataset.metadata[~filepath.map(lambda x: vad(pipeline, x))]

    dataset.metadata.to_csv("utils/filtered_metadata.csv")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)