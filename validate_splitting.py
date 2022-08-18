#! /home/schlechb/.conda/envs/mhciipresentation/bin/python
# -*- coding: utf-8 -*-

import itertools
import json
import multiprocessing as mp
from collections import Counter
from subprocess import call

import Levenshtein  # TODO: check if alternative dependency can be used to compute the distance
import pandas as pd

print("Loading Data")
sa_el_data = pd.read_csv(
    "~/PycharmProjects/data/mhciipresentation/preprocessed_debug.csv",
    index_col=0,
)
positive_peptides = list(
    sa_el_data["peptide"][sa_el_data["target_value"] == 1].values
)
X = dict()
X["train"] = pd.read_csv(
    "~/PycharmProjects/data/mhciipresentation/X_train.csv", index_col=0
)
X["test"] = pd.read_csv(
    "~/PycharmProjects/data/mhciipresentation/X_test.csv", index_col=0
)
X["val"] = pd.read_csv(
    "~/PycharmProjects/data/mhciipresentation/X_val.csv", index_col=0
)

pepset_dict = dict()
print("generating sets")
pepset_dict["test"] = set(X["test"]["peptide"])
pepset_dict["train"] = set(X["train"]["peptide"])
pepset_dict["val"] = set(X["val"]["peptide"])


def test_sub(str_a, str_b):
    if str_a in str_b or str_b in str_a:
        return True
    else:
        return False


def find_lev_sims(peptide, pep_list):
    lev_ratios = [Levenshtein.ratio(peptide, el) for el in pep_list]
    similars = list(pep_list[[el > 0.9 for el in lev_ratios]].values)

    return {peptide: similars}


def find_lev_sims_multiwrapper(args):
    return find_lev_sims(*args)


def calc_lev_ratio(peptide, pep_list):
    lev_ratios = [Levenshtein.ratio(peptide, el) for el in pep_list]
    return {peptide: lev_ratios}


def calc_lev_multiwrapper(args):
    return calc_lev_ratio(*args)


def test_pep(peptide, pep_list):
    matches = [test_sub(peptide, el) for el in pep_list]
    return pep_list[matches]


def test_multiwrapper(args):
    return test_pep(*args)


def assign_lev_group(peplist):
    peplist = list(set(peplist))
    newlist = list()
    count = 0
    moved_peps = 0
    while peplist:
        pep = peplist[0]
        count = count + 1
        if count % 100 == 0:
            print("peptide %s of %s" % (count, len(peplist)))
            print("moved %s peptides to new list" % moved_peps)
            moved_peps = 0
            print("moved %s peptides in total to new list" % len(newlist))
        lev_ratios = [Levenshtein.ratio(pep, el) for el in peplist]
        similars = [
            el for el, ratio in list(zip(peplist, lev_ratios)) if ratio > 0.9
        ]
        lev_group_id = Levenshtein.median(similars)
        moved_peps = moved_peps + len(similars)
        for similar in similars:
            newlist.append({"peptide": similar, "lev_group": lev_group_id})
            peplist.remove(similar)

    return newlist


# parallel execution of test
duplicate_peps = dict()
for pepset_name in ["val", "test"]:
    peplist = list()
    for pep in pepset_dict[pepset_name]:
        peplist.append((pep, X["train"]["peptide"]))

    print("start parallel execution")
    workers = mp.Pool(20)
    all_sims = workers.map(find_lev_sims_multiwrapper, peplist)

    for sims in all_sims:
        pep = list(sims)[0]
        if sims[pep]:
            print(
                "peptide %s of %s has similar peptides in train set"
                % (pep, pepset_name)
            )
            for sim_pep in sims[pep]:
                print(sim_pep)

    jsonfilename = "test_%s_similars.json" % pepset_name
    jsonfile = open(jsonfilename, "w")
    duplicate_peps[pepset_name] = [
        el for el in all_sims if not (list(el.values()) == [[]])
    ]
    json.dump(duplicate_peps[pepset_name], jsonfile)

    jsonfile.close()


duplicates_in_set = dict()
for pepset_name in ["val", "test"]:
    duplicates_in_set[pepset_name] = [
        list(el.keys())[0] for el in duplicate_peps[pepset_name]
    ]
    # duplicates_in_train[pepset_name] = list(
    #     itertools.chain(
    #         *[
    #             list(itertools.chain(*el.values()))
    #             for el in duplicate_peps[pepset_name]
    #         ]
    #     )
    # )  # TODO: where does duplicates_in_train come from?

    print(
        Counter(
            X[pepset_name][
                X[pepset_name]["peptide"].isin(duplicates_in_set[pepset_name])
            ]["target_value"]
        )
    )
    print(Counter(X[pepset_name]["target_value"]))
