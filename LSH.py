__author__ = 'Mike Starr, Matt Dennewitz, Mariusz Liksza (modifications)'
__copyright__ = "Copyright (c) 2014 Mike Starr"
__license__ = "MIT"


"""Algorithm based on 'Mining of Massive Datasets"""

from collections import defaultdict

from UnionFind import UnionFind


class Signature(object):
    """Signature Base class."""

    def __init__(self, dim):
        self.dim = dim
        self.hashes = self.hash_functions()

    def hash_functions(self):
        """Returns dim different hash functions"""
        pass

    def sign(self, object):
        """Return the signature for object s"""
        pass


class MinHashSignature(Signature):
    """Creates signatures for sets/tuples using minhash."""

    def hash_functions(self):
        """Return dim different hash functions"""
        def hash_factory(n):
            return lambda x: hash("salt" + unicode(n) + unicode(x) + "salt")
        return [ hash_factory(_) for _ in range(self.dim) ]

    def sign(self, s, k):
        """Returns minhash signature for set s"""
        s = shingle(s, k)
        sig = [ float("inf") ] * self.dim
        for hash_ix, hash_fn in enumerate(self.hashes):
            sig[hash_ix] = min(hash_fn(value) for value in s)
        return sig

class LSH(object):
    """Locality sensitive hashing.  Uses a banding approach to hash
    similar signatures to the same buckets."""
    def __init__(self, length, threshold):
        self.length = length
        self.threshold = threshold
        self.bandwidth = self.get_bandwidth(length, threshold)
    def hash(self, sig):
        """Generate hashvals for this signature"""
        for band in zip(*(iter(sig),) * self.bandwidth):
            yield hash("salt" + unicode(band) + "tlas")

    def get_bandwidth(self, n, t):
        """Approximates the bandwidth (number of rows in each band)
        needed to get threshold.

        Threshold t = (1/b) ** (1/r) where
        b = #bands
        r = #rows per band
        n = b * r = #elements in signature
        """
        best = n, 1
        minerr  = float("inf")
        for r in range(1, n + 1):
            try:
                b = 1. / (t ** r)
            except:             # Divide by zero, your signature is huge
                return best
            err = abs(n - b * r)
            if err < minerr:
                best = r
                minerr = err
        return best

    def get_threshold(self):
        r = self.bandwidth
        b = self.length / r
        return (1. / b) ** (1. / r)

    def get_n_bands(self):
        return int(self.length / self.bandwidth)


class Cluster(object):
    """Clusters sets with Jaccard similarity above threshold with high
    probability.

    Algorithm based on Rajaraman, "Mining of Massive Datasets":
    1. Generate set signature
    2. Use LSH to map similar signatures to same buckets
    3. Use UnionFind to merge buckets containing same values
    """
    # width - liczba sygnatur, k - dlugosc shingelsow
    def __init__(self, width, threshold, k):
        self.width = width
        self.unionfind = UnionFind()
        self.signer = MinHashSignature(width)
        self.hasher = LSH(width, threshold)
        self.hashmaps = [defaultdict(list)
                         for _ in range(self.hasher.get_n_bands())]
        self.k = k
    def add_set(self, s, label=None):
        """A label for this set"""
        if not label:
            label = s
        # Add to unionfind structure
        self.unionfind[label]
        # Get signature
        sig = self.signer.sign(s, self.k)
        # Union labels with same LSH key in same band
        for band_idx, hshval in enumerate(self.hasher.hash(sig)):
            self.hashmaps[band_idx][hshval].append(label)
            self.unionfind.union(label, self.hashmaps[band_idx][hshval][0])
    def get_sets(self):
        return self.unionfind.sets()

def shingle(s, k):
    """Generate k-length shingles of string s"""
    shingles = []
    k = min(len(s), k)
    for i in range(len(s) - k + 1):
        shingles.append(s[i:i+k])
    return shingles