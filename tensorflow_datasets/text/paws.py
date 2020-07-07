"""paws dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_datasets.public_api as tfds
import tensorflow as tf
from tensorflow_datasets.core.features import ClassLabel, Text
import os


# TODO(paws): BibTeX citation
_CITATION = """
@InProceedings{paws2019naacl,
  title = {{PAWS: Paraphrase Adversaries from Word Scrambling}},
  author = {Zhang, Yuan and Baldridge, Jason and He, Luheng},
  booktitle = {Proc. of NAACL},
  year = {2019}
}
"""

# TODO(paws):
_DESCRIPTION = """
"""


class Paws(tfds.core.GeneratorBasedBuilder):
  """TODO(paws): Short description of my dataset."""

  # TODO(paws): Set up version.
  VERSION = tfds.core.Version('0.1.0')

  def _info(self):

    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            'idx': tf.int32,
            'label': ClassLabel(num_classes=2),
            'sentence1': Text(),
            'sentence2': Text(),
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        # supervised_keys=("sentence1", "sentence2", "label"),
        # Homepage of the dataset for documentation
        homepage='https://github.com/google-research-datasets/paws',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""

    extracted_paths = dl_manager.download_and_extract({
        "wiki": "https://storage.googleapis.com/paws/english/paws_wiki_labeled_final.tar.gz",
    })

    wiki_path = extracted_paths["wiki"]

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={"dataset_path": os.path.join(wiki_path, "final", "train.tsv")},
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={"dataset_path": os.path.join(wiki_path, "final", "dev.tsv")},
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={"dataset_path": os.path.join(wiki_path, "final", "test.tsv")},
        ),
    ]

  def _generate_examples(self, dataset_path):
    """Yields examples."""

    with tf.io.gfile.GFile(dataset_path) as f:
        file_rows = f.readlines()[1:]

        for row in file_rows:
            records = row.split('\t')
            index, sentence_1, sentence_2, label = int(records[0]), records[1], records[2], int(records[3])
            yield index, {
                'idx': index,
                'label': label,
                'sentence1': sentence_1,
                'sentence2': sentence_2,
            }

