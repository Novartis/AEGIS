#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""runAEGIS.py

This is the CLI to run predictions
"""

import argparse
import os
import re
import sys

import pandas as pd
import torch
from Bio import SeqIO

from mhciipresentation.constants import AA_TO_INT, USE_GPU
from mhciipresentation.inference import setup_model_local
from mhciipresentation.loaders import load_pseudosequences
from mhciipresentation.paths import PROD_MODELS_DIR
from mhciipresentation.utils import (
    encode_aa_sequences,
    make_predictions_with_transformer,
    set_pandas_options,
)

cmd_folder = os.path.split(os.path.realpath(os.path.abspath(__file__)))[0]
sys.path.extend(
    [
        os.path.abspath(os.path.join(cmd_folder, "../../mhciipresentation")),
        os.path.abspath(os.path.join(cmd_folder, "../../../src")),
    ]
)


set_pandas_options()


class Parameters:
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", type=str, help="input file")

    parser.add_argument(
        "-m", "--modeltype", type=str, help="PAN or SPC model",
    )

    parser.add_argument(
        "-a", "--allele", type=str, default=None, help="allele(s)",
    )

    parser.add_argument(
        "-k",
        "--kmer",
        type=int,
        default=None,
        help="integer defining k-mer size",
    )

    parser.add_argument(
        "-o", "--output", type=str, help="path for output",
    )

    FLAGS = parser.parse_args()
    data = FLAGS.input
    modeltype = FLAGS.modeltype
    output = FLAGS.output
    alleles = FLAGS.allele
    kmer = FLAGS.kmer


def dataprep(data_file_name, alleles, kmer):
    def special_AA_conv(AA_seq) -> list:

        if "B" in AA_seq or "Z" in AA_seq:
            # B --> Q or D
            new_seqs = list(
                {AA_seq.replace("B", "Q", 1), AA_seq.replace("B", "D", 1)}
            )
            while bool([True for el in new_seqs if "B" in el]):
                new_Q = [el.replace("B", "Q", 1) for el in new_seqs]
                new_D = [el.replace("B", "D", 1) for el in new_seqs]
                new_seqs = new_Q + new_D

            # Z --> E or N
            while bool([True for el in new_seqs if "Z" in el]):
                new_E = [el.replace("Z", "E", 1) for el in new_seqs]
                new_N = [el.replace("Z", "N", 1) for el in new_seqs]
                new_seqs = new_E + new_N

        else:
            new_seqs = [AA_seq]

        return list(set(new_seqs))

    def seq2kmer(sequence, kmer) -> list:
        pep_list = list()
        for i in range(0, len(sequence) - kmer):
            pep_list.extend([sequence[i : i + kmer]])

        return pep_list

    check_peptide = re.compile("[ABCDEFGHIKLMNPQRSTVWYZ]*")
    peptides = list()
    if os.path.exists(data_file_name):
        handle = open(data_file_name, "r")
        firstline = handle.readline()
        if ">" in firstline:
            if not kmer:
                print("using default k-mer length k = 9")
                kmer = 9
            print(
                "we seem to have a fasta file. Preparing peptide k-mers (k = %i) using sliding window"
                % kmer
            )
            # we have probably a fasta file
            handle.close()
            record_dict = SeqIO.to_dict(SeqIO.parse(data_file_name, "fasta"))
            for record_key in record_dict.keys():
                sequence = str(record_dict[record_key].seq)
                peptides.extend(seq2kmer(sequence, kmer))

        elif check_peptide.fullmatch(firstline.strip()):
            # we have a peptide list
            print("We seem to have a txt file with peptides")
            if not kmer:
                print("Using peptides as they appear.")
            peptides.extend(special_AA_conv(firstline.strip()))
            for line in handle.readlines():
                if check_peptide.fullmatch(line.strip()):
                    peptides.extend(special_AA_conv(line.strip()))
                else:
                    print(
                        "peptide %s contains invalid AA letters" % line.strip()
                    )

            if kmer:
                print("generating k-mers (k=%i) if needed" % kmer)

                new_peplist = list()
                for pep in peptides:
                    if len(pep) > kmer:
                        new_peplist.extend(seq2kmer(pep, kmer))
                    else:
                        new_peplist.extend([pep])
                peptides = new_peplist

    else:
        # Is it a peptide string?"
        if check_peptide.fullmatch(data_file_name):
            peptides.extend(special_AA_conv(data_file_name))

    if alleles:
        # load allele Pseudosequence mapping
        pseudoseqs = load_pseudosequences()
        pseudoseqs_dict = {
            el["Name"]: el["Pseudosequence"]
            for el in pseudoseqs.to_dict("records")
        }

        allele_list = list()
        if os.path.exists(alleles):
            handle = open(alleles, "r")
            for line in handle.readlines():
                allele_list.extend([line.strip()])
        else:
            print("WARNING: allele file %s does not exist." % alleles)

        pseudoseqs_list = list()
        for allele in allele_list:
            if allele in pseudoseqs_dict.keys():
                pseudoseqs_list.append((allele, pseudoseqs_dict[allele]))

        # create permutation of peptides with pseudosequences

        input_records = list()
        for pep in peptides:
            for allele, pseudoseq in pseudoseqs_list:
                input_records.append(
                    {
                        "Sequence": pep,
                        "MHC_molecule": allele,
                        "Pseudoseqence": pseudoseq,
                        "peptides_and_pseudosequence": pep + pseudoseq,
                    }
                )

        return pd.DataFrame.from_records(input_records)

    else:
        input_records = list()
        for pep in peptides:
            input_records.append(
                {"Sequence": pep, "peptides_and_pseudosequence": pep}
            )

        return pd.DataFrame.from_records(input_records)


def main():
    # get params from CLI
    path2data = Parameters.data
    modeltype = Parameters.modeltype
    alleles = Parameters.alleles
    kmer = Parameters.kmer
    outputpath = Parameters.output

    # parse and prep data
    input_data = dataprep(path2data, alleles, kmer)
    X = encode_aa_sequences(input_data.peptides_and_pseudosequence, AA_TO_INT,)

    # set up model and pytorch device
    models_dict = {
        "SPC": os.path.join(
            PROD_MODELS_DIR, "transformer/SPC_public/best_model.pth"
        ),
        "PAN": os.path.join(
            PROD_MODELS_DIR, "transformer/PAN_public/best_model.pth"
        ),
    }

    MODELPATH = models_dict[modeltype]

    batch_size = 5000
    device = torch.device("cuda" if USE_GPU else "cpu")  # training device
    model, input_dim = setup_model_local(device, MODELPATH)

    # run prediction
    predictions = make_predictions_with_transformer(
        X, batch_size, device, model, input_dim, AA_TO_INT["X"]
    )

    print("")
    print("done with prediction")
    # store prediction
    output_data = input_data.assign(prediction=predictions)
    print("writing outputdata to %s" % outputpath)
    output_data.to_csv(outputpath)

    print("finished")


if __name__ == "__main__":
    main()
