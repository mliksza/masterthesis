__author__ = 'Mariusz Liksza'


def create_dictionary_from_clusters(file_name, dictionary_urls):
    """
    @type file_name: str
    @type dictionary_urls: dict
    """
    dict_variables = dict()
    counter = 0
    with open(file_name) as clusters:
        for line in clusters:
            if line != "\n":
                # if cluster number -> increase the value of a counter
                if line.rstrip() == str(counter):
                    counter += 1
                else:
                    # if URL address, replace it on appropriate numerical value
                    # using previously prepared dictionary, then assign each URL
                    # (in numerical format) to the group number
                    dict_variables[dictionary_urls[line.rstrip()]] = counter
            else:
                pass
    print "The dictionary has been created."
    return dict_variables