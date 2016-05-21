__author__ = 'Mariusz Liksza'

from LSH import Cluster

def test_similar_sets(lsh_input_file_name, lsh_output_dir,
                      signature, threshold, length_of_k_shingles):
    """
    @type lsh_input_file_name: str
    @type lsh_output_dir: str
    @type signature list: list
    @type threshold list: list
    @type length_of_k_shingles: list
    """
    cluster = Cluster(signature, threshold, length_of_k_shingles)

    with open(lsh_input_file_name) as url_file:
        for line in url_file:
            cluster.add_set(line.strip())

    LSHoutput_file_whole_name = "{0}{1}_{2}_{3}.txt"\
        .format(lsh_output_dir, str(signature),
                str(threshold), str(length_of_k_shingles))

    f = open(LSHoutput_file_whole_name, "w")
    for cluster_number, cluster_urls in enumerate(cluster.get_sets()):
        processed_cluster_urls = str(cluster_urls)\
                                    .replace("[", "")\
                                    .replace("]", "")\
                                    .replace("', ", "\n")\
                                    .replace("'", "")
        print >> f, "{0}\n{1}\n".format(cluster_number, processed_cluster_urls)

    print "Grouped URLs has been saved in the file: " + LSHoutput_file_whole_name
    return LSHoutput_file_whole_name