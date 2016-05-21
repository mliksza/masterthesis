__author__ = 'Mariusz Liksza'

import sys

reload(sys)
sys.setdefaultencoding("utf-8")

# import necessary libraries
import json
import unicodedata
import os
import csv

from urlparse import urlparse
from collections import Counter
from sklearn.cross_validation import train_test_split
import itertools as it

# import own modules
import TestCluster
from TestCluster import test_similar_sets
from CreateVarFromGroups import create_dictionary_from_clusters

# import data from json file and return as json_data
def import_data(file_name):
    """
    @type file_name: str
    """
    json_data = list()
    with open(file_name) as json_file:
        for line in json_file:
            json_data.append(json.loads(line))
        print "Data: " + file_name + " has been loaded."
    return json_data

# clean, prepare and create variables from imported data (json_data) using appropriate method,
# parameter method indicates specific approach to create variable from URLs,
# method = 1 - take whole URLs, method = 2 - take domains + 1. catalog,
def write_data_to_var(data, method):
    """
    @type data: list
    @type method: int
    """
    grouped_urls = list()
    counter_urls = list()
    user_ids = list()
    labels = list()
    views = list()
    unique_url_numbers = list()
    dict_urls = dict()
    dict_urls_domains = dict()
    url_counter = 1
    weights = list()

    for user_data in range(len(data)):
        url_numbers = list()
        user_agent = data[user_data]["userAgent"]
        view = data[user_data]["views"]

        # remove bots and those users who visited pages too often (outliers)
        if ("Googlebot" in user_agent or view > 50):
            continue

        try:
            # check whether the user clicked on pop-up or not
            for user_action in range(len(data[user_data]["actions"])):
                action = data[user_data]["actions"][user_action]["action"]
                if action == "popupClicked":
                    action = 1
                    weight = 34  # 34 more -1 than 1
                    break
                else:
                    action = -1  # -1 instead 0 - specific for Vowpal Wabbit
                    weight = 1
            if method == 1:  # create variable from whole URLs
                for user_url in range(len(data[user_data]["urls"])):
                    url = unicodedata.normalize("NFD", data[user_data]["urls"][user_url]["url"]). \
                        replace(u"ł", "l").encode("ascii", "ignore")
                    parsed_url = urlparse(url)
                    tmp_url = parsed_url.netloc + parsed_url.path
                    truncated_url = tmp_url.split(",", 1)[0].split("RD", 1)[0]
                    cleaned_url = truncated_url.translate(None, "!@#$.-/_")
                    # create dictionary in order to assign values to URLs (for performance objectives)
                    if cleaned_url != "":
                        if cleaned_url not in dict_urls:
                            dict_urls[cleaned_url] = url_counter
                            url_counter += 1
                            unique_url_numbers.append(cleaned_url)
                        url_numbers.append(dict_urls[cleaned_url])
            elif method == 2:  # create variable from domains + 1. catalog
                for user_url in range(len(data[user_data]["urls"])):
                    url  = unicodedata.normalize("NFD", data[user_data]["urls"][user_url]["url"]). \
                        replace(u"ł", "l").encode("ascii", "ignore")
                    parsed_url = urlparse(url)
                    tmp_url = parsed_url.netloc + parsed_url.path
                    splitted_url = tmp_url.split("/")
                    try:
                        url_part = splitted_url[0] + splitted_url[1]
                    except IndexError:
                        url_part = splitted_url[0]
                    url_part_cleaned = url_part.translate(None, "!@#$.-/_")
                    if url_part_cleaned != "":
                        if url_part_cleaned not in dict_urls_domains:
                            dict_urls_domains[url_part_cleaned] = url_counter
                            url_counter += 1
                            unique_url_numbers.append(url_part_cleaned)
                        url_numbers.append(dict_urls_domains[url_part_cleaned])
            labels.append(action) # add labels (user clicked or not)
            weights.append(weight)  # add weights
            user_ids.append(data[user_data]["uid"].encode("ascii")) # add user ids
            views.append(view) # add views
            grouped_urls.append(url_numbers) # add URLs grouped by users
            counter_urls.append(Counter(url_numbers)) # unique URLs with the number of visited pages
                                                      # grouped by user
        except KeyError:
            pass
    print "Data has been assigned to the variables."
    return grouped_urls, counter_urls, labels, weights, \
           user_ids, views, unique_url_numbers, dict_urls


# write prepared URLs (whole URLs or domains + 1. catalog) to file
def write_to_file(file_name, urls):
    """
    @type file_name: str
    @type urls: list
    """
    with open(file_name, "w") as f:
        f.write("\n".join(urls))
    print "The URLs has been saved to the file: " + file_name


# create new variables (only if we use before test_similar_sets function from TestCluster.py
# and create_dictionary_from_clusters from CreateVarFromGroups.py)
def create_variables_from_LSH(urls, dictionary):
    """
    @type urls: list
    @type dictionary: dict
    """
    counter_urls = list()
    for user_urls in urls:
        new_urls = list()
        for url in user_urls:
            new_urls.append(dictionary[url])
        counter_urls.append(Counter(new_urls))
    print "New variables has been created (based on grouped URLs)."
    return counter_urls


# prepare data to Vowpal Wabbit format
def prepare_data_to_vw(urls, users, actions, importance):
    """
    @type urls: list
    @type users: list
    @type actions: list
    @type importance: list
    """
    # independent variables (URLs)
    variables_tovw = list()
    for user_counter_urls in urls:
        variables_tovw.append(str(user_counter_urls)
                              .replace("Counter({", "")
                              .replace(",", "")
                              .replace("})", ""))
    # user ids, "|" separates the features from classes, weights and user ids
    user_ids_tovw = [str(user) + "|" for user in users]
    # dependent variable (clicked/not clicked)
    labels_tovw = [str(action) for action in actions]
    # weights, 34 times greater for rare class (i.e. 1)
    weights_tovw = [str(weight) for weight in importance]
    # merge above lists
    merged = zip(labels_tovw, weights_tovw, user_ids_tovw, variables_tovw)
    vw_data = [" ".join(set) for set in merged]
    print "Data has been prepared to the Vowpal Wabbit."
    return vw_data


# split prepared data on the training and test set
def split_data(data, train_file_name, test_file_name):
    """
    @type data: list
    @type train_file_name: str
    @type test_file_name: str
    """
    # random_state=0 - always the same separation
    data_train, data_test = train_test_split(data, test_size=0.3, random_state=0)
    with open(train_file_name, "w") as g:
        for line in data_train:
            g.write(line + "\n")
    with open(test_file_name, "w") as g:
        for line in data_test:
            g.write(line + "\n")
    print "Training set has been saved to the file: " + train_file_name + ", and test set to the file:  " + test_file_name


#######################################
### The whole process (without LSH) ###
#######################################

def whole_process(method, cleaned_urls_file_name,
                  output_train_file_name, output_test_file_name):
    """
    @type method: int
    @type cleaned_urls_file_name: str
    @type output_train_file_name: str
    @type output_test_file_name: str
    """
    grouped_urls, counter_urls, user_ids, labels, weights, \
    views, unique_url_numbers, dict_urls = \
        write_data_to_var(data=json_data, method=method)
    write_to_file(file_name=cleaned_urls_file_name, urls=unique_url_numbers)
    vw_data = prepare_data_to_vw(urls=counter_urls, users=user_ids,
                                 actions=labels, importance=weights)
    split_data(vw_data, train_file_name=output_train_file_name,
               test_file_name=output_test_file_name)
    print "The process has been completed."


# import data and call above final method
json_data = import_data(file_name="export_27012015.json")
if __name__ == "__main__":
    whole_process(method=1,
                  cleaned_urls_file_name="cleaned_urls.txt",
                  output_train_file_name="cleaned_urls_train.txt",
                  output_test_file_name="cleaned_urls_test.txt")


# build model: Vowpal Wabbit (stochastic gradient descent method)
def build_model(train_file_name, test_file_name, predictions_file_name,
                labels_file_name, model_name):
    train_model = "vw --data={0} --loss_function=logistic --link=logistic \
                   --readable_model={1} --final_regressor=model.vw" \
                  .format(train_file_name, model_name)

    make_predictions = "vw --data={0} --testonly --initial_regressor=model.vw \
                        --loss_function=logistic --predictions={1}"\
                       .format(test_file_name, predictions_file_name)

    labels_test_set = "cut -d ' ' -f 1-3 {0} | sed -e 's/^-1/0/' > {1}"\
                      .format(test_file_name, labels_file_name)

    os.system(train_model)
    os.system(make_predictions)
    os.system(labels_test_set)


build_model(train_file_name='whole_urls_train.txt',
            test_file_name='whole_urls_test.txt',
            predictions_file_name='whole_urls_predictions.txt',
            labels_file_name="whole_urls_labels.txt",
            model_name="whole_urls_model.txt")



########################################
### The whole process: including LSH ###
########################################

# make all combinations from LSH parameters (write everything as document)
def make_lsh_documentation(doc_file_name, lsh_output_file_name, cleaned_urls_file_name,
                           output_train_file_name, output_test_file_name, **kwargs):
    key_names = sorted(kwargs)
    combinations = [dict(zip(key_names, prod))
                    for prod in it.product(*(kwargs[key] for key in key_names))]
    meta_data = dict()
    meta_data["The name of the file with cleaned URLs: "] = cleaned_urls_file_name
    meta_data["The first part of the LSH results file name: "] = lsh_output_file_name
    meta_data["The first part of the file with the training set: "] = output_train_file_name
    meta_data["The first part of the file with the test set: "] = output_test_file_name
    with open(doc_file_name, "w") as file:
        file.write(str(meta_data) + "\n")
        for line in combinations:
            file.write(str(line) + "\n")
    return meta_data, combinations


meta_data, combinations = make_lsh_documentation(
    doc_file_name="documentation_20160222.txt",
    cleaned_urls_file_name="cleaned_urls.txt",
    lsh_output_file_name="LSHoutput",
    output_train_file_name="cleaned_urls_train",
    output_test_file_name="cleaned_urls_test",
    signature=[100, 150, 200, 250],  # you can provide any number of values
    threshold=[0.5, 0.6, 0.7, 0.8],  # you can provide any number of values
    k_shingle=[4, 5, 6, 7, 8]        # you can provide any number of values
    )


def whole_process_lsh():
    grouped_urls, counter_urls, user_ids, labels, weights, \
    views, unique_url_numbers, dict_urls \
        = write_data_to_var(data=json_data, method=1)
    cleaned_urls_file_name = meta_data["The name of the file with cleaned URLs: "]
    lsh_output_file_name = meta_data["The first part of the LSH results file name: "]
    output_train_file_name = meta_data["The first part of the file with the training set: "]
    output_test_file_name = meta_data["The first part of the file with the test set: "]
    write_to_file(file_name=cleaned_urls_file_name, urls=unique_url_numbers)

    for line in combinations:
        LSHoutput_file_whole_name = test_similar_sets(
            lsh_input_file_name=cleaned_urls_file_name,
            lsh_output_dir=lsh_output_file_name,
            signature=line["signature"],
            threshold=line["threshold"],
            length_of_k_shingles=line["k_shingle"])

        dict_variables = create_dictionary_from_clusters(
            file_name=LSHoutput_file_whole_name,
            dictionary_urls=dict_urls)

        counter_urls = create_variables_from_LSH(
            urls=grouped_urls, dictionary=dict_variables)

        vw_data = prepare_data_to_vw(
            urls=counter_urls, users=user_ids,
            actions=labels, importance=weights)

        output_train_whole_file_name = "{0}_{1}_{2}_{3}.txt" \
            .format(output_train_file_name, str(line["signature"]),
                    str(line["threshold"]), str(line["k_shingle"]))

        output_test_whole_file_name = "{0}_{1}_{2}_{3}.txt" \
            .format(output_test_file_name, str(line["signature"]),
                    str(line["threshold"]), str(line["k_shingle"]))

        split_data(vw_data, train_file_name=output_train_whole_file_name,
                   test_file_name=output_test_whole_file_name)

    print "The process has been completed."


# import data and call above final method (including grouping URLs using LSH)
json_data = import_data(file_name="export_27012015.json")
if __name__ == "__main__":
    whole_process_lsh()


# build model for every combination of parameters:
# Vowpal Wabbit (stochastic gradient descent method)
os.chdir("LSH_20160220")
for line in combinations:
    train_file_name = "{0}_{1}_{2}_{3}.txt"\
        .format("cleaned_urls_train", str(line["signature"]),
                str(line["threshold"]), line["k_shingle"])

    test_file_name = "{0}_{1}_{2}_{3}.txt"\
        .format("cleaned_urls_test", str(line["signature"]),
                str(line["threshold"]), line["k_shingle"])

    predictions_file_name = "{0}_{1}_{2}_{3}.txt"\
        .format("LSH_predictions", str(line["signature"]),
                str(line["threshold"]), line["k_shingle"])

    labels_file_name = "{0}_{1}_{2}_{3}.txt"\
        .format("LSH_classes", str(line["signature"]),
                str(line["threshold"]), line["k_shingle"])

    model_name = "{0}_{1}_{2}_{3}.txt"\
        .format("LSH_model", str(line["signature"]),
                str(line["threshold"]), line["k_shingle"])

    build_model(train_file_name, test_file_name,
                predictions_file_name, labels_file_name, model_name)