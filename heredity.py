import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    
    joint_p = 1

    for person, family in people.items():

        mother = family["mother"]
        father = family ["father"]
        
        gene = 2 if person in two_genes else 1 if person in one_gene else 0
        trait = True if person in have_trait else False
        
        trait_prob = PROBS["trait"][gene][trait]
        gene_prob = 0

        if mother is None and father is None:
            gene_prob = PROBS["gene"][gene]
            
        else:
            mother_genes = 2 if mother in two_genes else 1 if mother in one_gene else 0
            father_genes = 2 if father in two_genes else 1 if father in one_gene else 0

            if gene == 0:
                # Get no genes from both parents
                gene_prob_mother = get_parent_gene_probability(False, mother, mother_genes)
                gene_prob_father = get_parent_gene_probability(False, father, father_genes)

                gene_prob = gene_prob_mother * gene_prob_father

            elif gene == 1:
                # Either they get the gene from their mother 
                # and not their father, or they get the gene 
                # from their father and not their mother.
                gene_prob_mother = get_parent_gene_probability(True, mother, mother_genes) * get_parent_gene_probability(False, father, father_genes)
                gene_prob_father = get_parent_gene_probability(False, mother, mother_genes) * get_parent_gene_probability(True, father, father_genes)

                # These two scenarios are mutually exclusive (only one or the other can happen), so we add:
                gene_prob = gene_prob_mother + gene_prob_father                

            elif gene == 2:
                # Get genes from both parents
                gene_prob_mother = get_parent_gene_probability(True, mother, mother_genes)
                gene_prob_father = get_parent_gene_probability(True, father, father_genes)

                gene_prob = gene_prob_mother * gene_prob_father            
        
        joint_p *= gene_prob * trait_prob
    
    return joint_p


def get_parent_gene_probability(child_has_gene, parent, parent_genes):
    """
    Returns the probability that a child inherits or does not inherit the gene from a parent.
    """
    if parent is not None:
        if child_has_gene == True:
            if parent_genes == 0:
                return PROBS["mutation"]
            elif parent_genes == 1:
                # Can pass the gene or can pass by mutation
                return 0.5 * (1 - PROBS["mutation"]) + 0.5 * PROBS["mutation"]
            elif parent_genes == 2:
                return 1 - PROBS["mutation"]
        else:
            if parent_genes == 0:
                return 1 - PROBS["mutation"]
            elif parent_genes == 1:
                # Not pass the gene or not pass the gene by mutation
                return 0.5 * (1 - PROBS["mutation"]) + 0.5 * PROBS["mutation"]
            elif parent_genes == 2:
                return PROBS["mutation"]
    else:
        return get_unknown_parent_gene_probability(child_has_gene)         


def get_unknown_parent_gene_probability(child_has_gene):
    """
    Returns the probability that the child received genes from unknown parent
    """
    if child_has_gene == True:
        gene_prob = PROBS["gene"][0] * PROBS["mutation"]
        gene_prob += PROBS["gene"][1] * (0.5 * (1 - PROBS["mutation"]) + 0.5 * PROBS["mutation"])
        gene_prob += PROBS["gene"][2] * (1 - PROBS["mutation"])

        return gene_prob
    
    else:
        gene_prob = PROBS["gene"][0] * (1 - PROBS["mutation"])
        gene_prob += PROBS["gene"][1] * (0.5 * (1 - PROBS["mutation"]) + 0.5 * PROBS["mutation"])
        gene_prob += PROBS["gene"][2] * (PROBS["mutation"])
        
        return gene_prob  
        

def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    
    for person in probabilities:
        gene = 2 if person in two_genes else 1 if person in one_gene else 0
        trait = True if person in have_trait else False

        probabilities[person]["gene"][gene] += p
        probabilities[person]["trait"][trait] += p
    

def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """

    for person, features in probabilities.items():        
        for feature, distribution in features.items():
            s = sum(distribution.values())
            for key, value in distribution.items():
                probabilities[person][feature][key] = value / s 


if __name__ == "__main__":
    main()
