import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Optional, List
import logging
import tqdm
import pandas as pd


@dataclass
class NoiseSegment:
    """Class for a noise segment in AudioSet"""
    noise_id: str
    labels: List[str]
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start
    
    @property
    def filename(self) -> str:
        return f"{self.noise_id}_{int(self.start*1000)}_{int(self.end*1000)}.flac"
    
    @property
    def label_names(self) -> List[str]:
        pass


@dataclass
class AudioSet:
    """Class for AudioSet"""
    metadata_path: Path
    label_path: Path = None
    # metadata: pd.DataFrame = None

    def __post_init__(self):
        self.metadata = pd.read_csv(
            self.metadata_path,
            sep=',\s+',
            skiprows=3,
            names=['file_id', 'start', 'end', 'labels'],
            converters={'labels': eval},
            engine='python'
        )
        self.metadata.labels = self.metadata.labels.str.split(",")

        if self.label_path is not None:
            df = pd.read_csv(self.label_path)
            label_name = df.set_index('mid')['display_name'].to_dict()
            self.metadata.labels = self.metadata.labels.apply(label_name)

    def __iter__(self) -> Iterable[NoiseSegment]:
        for index, segment in tqdm(self.metadata.iterrows(), total=self.metadata.shape[0]):
            noise_id, start, end, labels = segment
            noise_segment = NoiseSegment(noise_id, start, end, labels)
            yield noise_segment

    @property
    def filename(self) -> pd.Series:
        return self.metadata.apply(
            lambda row: f"{row['file_id']}_{int(row['start']*1000)}_{int(row['end']*1000)}.flac", axis=1
        )
    
    def filepath(self, data_path: Path) -> pd.Series:
        return self.filename.map(
            lambda x: data_path / x
        )

    def drop_files_not_found(self, data_path: Path) -> None:
        file_path_series = self.filepath(data_path)
        self.metadata = self.metadata[file_path_series.map(lambda x : x.exists())]