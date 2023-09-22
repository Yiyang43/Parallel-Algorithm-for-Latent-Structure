from __future__ import annotations
from collections import deque
from CrossCoverTester import CrossCoverTester
from LatentGroups import LatentGroups, allRanksEqual
from Cover import setLength, setDifference, Cover, pairwiseOverlap
import misc as M
from misc import Independences, Edges
from GraphDrawer import printGraph
from CCARankTester import CCARankTester
from itertools import combinations
from copy import deepcopy
import itertools
import logging
from scenarios import scenarios as SCENARIOS
from pdb import set_trace

LOGGER = logging.getLogger(__name__)


class IL2HGraphFinder:
    def __init__(self, alpha=0.05, maxk=3, sample=False, df=None, n=500):
        self.alpha = alpha  # Critical value for testing
        self.maxk = maxk
        self.sample = sample
        self.covList = []
        self.i = 1
        self.df = df
        self.n = n

    def runScenario(self, scenario: str, stage: int = 1):
        """
        Run the procedure on a user-defined scenario, assuming that we have
        access to the true covariance matrix between all measures.
        """
        LOGGER.info(f"Searching Scenario {scenario}...")
        self.g = SCENARIOS[scenario]()
        self.G = LatentGroups(self.g.xvars)

        # Create sample data
        if self.sample:
            df = self.g.generateData(n=self.n)
            self.addSample(df)

        if stage >= 1:
            self.findClusters()
            G = self.cleanup(self.G)
            printGraph(G, f"plots/scenario{scenario}_phase1.png")

        if stage >= 2:
            self.refineClusters()
            self.G = self.cleanup(self.G)
            printGraph(self.G, f"plots/scenario{scenario}_phase2.png")

        if stage >= 3:
            I, O, E = self.refineEdges()
            printGraph(E, f"plots/scenario{scenario}_phase3.png")

    def runSample(self, df, stage: int = 1):
        """
        Runs the procedure on sample data provided in df.

        Note that all variable names in df must start with "X" to indicate that
        they are measures.
        """
        for col in df.columns:
            assert col.startswith("X"), "Variable names must start with X."
        self.addSample(df=df)
        self.G = LatentGroups(df.columns)
        if stage >= 1:
            self.findClusters()
            G = self.cleanup(self.G)
            printGraph(G, f"plots/phase1.png")

        if stage >= 2:
            self.refineClusters()
            self.G = self.cleanup(self.G)
            printGraph(self.G, f"plots/phase2.png")

        if stage >= 3:
            I, O, E = self.refineEdges()
            printGraph(E, f"plots/phase3.png")

    def cleanup(self, G: LatentGroups):
        """
        Clean up a LatentGroups object by removing all temporary root variables
        and connecting their adjacent nodes in a chain structure. This is to
        ensure that the graph ends up with the correct number of latents.
        """
        G = deepcopy(G)
        tempRoots = [L for L in G.latentDict if L.isTemp]
        LOGGER.info(f"{'-'*15} Cleaning up ... {'-'*15}")
        for L in tempRoots:
            G.makeRoot(L)
            Vs = G.findChildren(L)

            # Check if we can remove the temp root without affecting rank tests
            if not pairwiseOverlap(Vs):
                Gp = deepcopy(G)
                Gp.removeCover(L)
                Gp.connectVariablesChain(Vs)
                if allRanksEqual(G, Gp, Vs):
                    # Connect variables in chain
                    LOGGER.info(f"Removing temp root {L}..")
                    G = Gp
                    continue

            #  Otherwise, convert temporary root into permanent variable
            LOGGER.info(f"Converting temp root {L} into permanent..")
            L.temp = False
            v = G.latentDict[L]
            G.latentDict.pop(L)
            G.latentDict[L] = v

        for L in G.latentDict:
            assert not L.isTemp, "Did not successfully clean up temp roots."

        M.display(G)
        printGraph(G)
        assert len(G.activeSet) == 1, "The graph should have one root variable."
        return G

    #################### Phase I: findClusters ###########################

    def findClusters(self):
        M.clearOutputFolder()
        G = deepcopy(self.G)  # Prevent modification of self.G
        self.G, _ = self._findClusters(G)

    def _findClusters(self, G: LatentGroups):
        """
        Called by findClusters and refineClusters.
        """
        assert isinstance(G, LatentGroups), "G must be a LatentGroups object."

        prevCovers = set(G.latentDict.keys())  # Record current latent Covers

        k = 1
        while True:
            LOGGER.info(f"{'-'*15} Test Cardinality now k={k} {'-'*15}")
            LPowerSet = self.generateLatentPowerset(G)
            activeSetCopy = deepcopy(G.activeSet)

            # Select a combination of latents, and replace their place in
            # the activeSet with their children for the search
            for Ls in LPowerSet:
                Vprime = deepcopy(activeSetCopy)
                for L in Ls:
                    children = G.findChildren(L)

                    # If children of L is just one variable, do not replace
                    if len(children) > 1:
                        Vprime = Vprime - set([L])
                        Vprime |= children
                G.activeSet = Vprime

                # After searchClusters, either:
                # 1. Some cluster found, reset k=1 and search again
                # 2. No cluster found and no vars remaining, so terminate
                # 3. No cluster found but some vars remaining, k+=1 and cont.
                G, (found, terminate) = self.searchClusters(G, k)

                if found:
                    # We have to reconnect nonAtomics every iteration so that
                    # they can be used in the findCluster process
                    G.reconnectNonAtomics()
                    G.updateActiveSet()

                # CASE 2
                if terminate:
                    break

                # CASE 1
                if found:
                    k = 1
                    break

            # CASE 2
            if (k > self.maxk) or terminate:
                LOGGER.info(f"Procedure ending...")
                break

            # CASE 3
            if not found:
                LOGGER.info("Nothing found!")
                k += 1

        G.updateActiveSet()
        G = self.finish(G)
        newCovers = set(G.latentDict.keys()) - prevCovers
        return G, newCovers

    def searchClusters(self, G: LatentGroups, k):
        """
        Run one round of search for clusters of size k.
        """
        LOGGER.info(f"Starting searchClusters k={k}...")

        found, terminate = self._searchClusters(G, k)

        # Procedure ends when there are no further variables to test
        if terminate:
            printGraph(G)
            return G, (found, terminate)

        if found:
            G.determineClusters()
            success = G.confirmClusters()

            # If no clusters successfully added, found becomes False
            # This can occur if the new clusters contradict previous ones
            if not success:
                found = False
            else:
                G.updateActiveSet()
                M.display(G)
                printGraph(G)

        return G, (found, terminate)

    def _searchClusters(self, G: LatentGroups, k=1):
        """
        Internal method for searchClusters.
        """
        terminate = False  # Whether we ran out of variables to test
        found = False  # Whether we found any clusters

        # Terminate if not enough active variables
        # To test, we need n >= 2k+2
        # So terminate if  n < 2k+2
        # i.e.             k > n/2 - 1
        if k > setLength(G.activeSet) / 2 - 1:
            terminate = True
            return (found, terminate)

        allSubsets = M.generateSubsetMinimal(G.activeSet, k)

        # If no subsets can be generated, terminate
        if allSubsets == [set()]:
            terminate = True
            return (found, terminate)

        for As in allSubsets:
            As = set(As)  # test set
            Bs = setDifference(G.activeSet, As)  # control set
            if setLength(As) > setLength(Bs):
                continue

            # As must not contain more than k elements from
            # any atomic Cover with cardinality <= k-1
            if G.containsCluster(As):
                continue

            # Bs parentCardinality cannot be < k+1, since otherwise
            # we get rank <= k regardless of what As is
            if G.parentCardinality(Bs) < k + 1:
                continue

            test, rk = self.structuralRankTest(G, As, Bs, k)
            if test:
                LOGGER.info(f"   {As} is rank deficient!")
                G.addRankDefSet(As, rk)
                found = True

        return (found, terminate)

    def generateLatentPowerset(self, G):
        """
        Generate an iterator over powerset of active latents,
        Including the combination where all latents are included.
        """
        Ls = set([V for V in G.activeSet if V.isLatent])
        Lsubsets = reversed(list(M.powerset(Ls)))
        # if len(Ls) > 0:
        #    next(iter(Lsubsets)) # discard first entry
        return Lsubsets

    def finish(self, G: LatentGroups):
        """
        Procedure for completing the graph when no more clusters may be found,
        by introducing a temporary root latent variable.
        """
        LOGGER.info(f"{'-'*15} Introducing temporary root variable ... {'-'*15}")
        if len(G.activeSet) == 1:
            pass
        elif len(G.activeSet) > 2:
            G.introduceTempRoot()
        elif len(G.activeSet) == 2:
            root, child = list(G.activeSet)
            if len(root.vars.intersection(child.vars)) > 0:
                G.introduceTempRoot()
            else:
                if root.isLatent:
                    G.addOrUpdateCover(root, set([child]))
                elif child.isLatent:
                    G.addOrUpdateCover(child, set([root]))
                else:
                    G.introduceTempRoot()
                G.updateActiveSet()
        M.display(G)
        printGraph(G)
        assert len(G.activeSet) == 1, "The graph should have one root variable."
        return G

    ######################## Phase II: refineClusters ######################
    def refineClusters(self):
        LOGGER.info(f"{'='*10} Refining Clusters! {'='*10}")

        # Set all refined to False
        for L in self.G.latentDict:
            self.G.latentDict[L]["refined"] = False

        # Start off Breadth-First-Search from root
        Q = deque()  # FIFO queue for BFS
        root = next(iter(self.G.activeSet))
        for C in M.reorderCovers(self.G.findChildren(root)):
            if C.isLatent:
                Q.append(C)

        Gp = deepcopy(self.G)
        while len(Q) > 0:
            L = Q.popleft()  # pop earliest appended item

            if (not L in Gp.latentDict) or Gp.isRefined(L):
                continue

            # Add children to Q if not refined
            children = Gp.findChildren(L)
            for C in children:
                if C.isLatent and not (Gp.isRefined(C)) and (C not in Q):
                    Q.append(C)
                    LOGGER.info(f"Appending child {C} to queue..")

            LOGGER.info(f"Going to refine {L}...")
            Gp = self.refineCover(Gp, L)

            LOGGER.info(f"Refined {L}...")
            M.display(Gp)
            printGraph(Gp)

        LOGGER.info(f"{'='*20} Finished Refining! {'='*20}")
        self.G = Gp
        M.display(self.G)
        printGraph(self.G)

    def refineCover(self, Gp, L):
        M.display(Gp)
        printGraph(Gp)

        # Remove single-child linkages
        # Use copy to avoid "set changed size during iteration" error
        children = Gp.findChildren(L).copy()
        if len(children) == 1:
            LOGGER.info(f"Connecting {L}'s parent to {L}'s children")
            Gp.bypassSingleChild(L)

        Gp.dissolveNode(L)
        M.display(Gp)
        printGraph(Gp)

        Gp, newLs = self._findClusters(Gp)

        # Set newly discovered latents as refined
        for newL in newLs:
            Gp.latentDict[newL]["refined"] = True
        LOGGER.info(f"Setting {newLs} as refined...")
        return Gp

    ### Phase III: refineEdges ###
    def refineEdges(self):
        I = Independences()
        E = Edges()
        O = set()

        # Find I from non-atomics first as they can affect all localities
        I_nonatomics = Independences()
        for L in self.G.latentDict:
            if L.isAtomic:
                continue
            I_nonatomics.update(self.findIndependenceForNonAtomic(L))
        I.update(I_nonatomics)

        # Then, get independence at the locality of each atomic Cover
        for L in self.G.latentDict:
            if not (L.isAtomic and not L.isTemp):
                continue

            I_internal = deepcopy(I_nonatomics)
            I_temp, partitions = self.findIndependenceAcrossPartitions(L)
            I_internal.update(I_temp)

            Vs = set([L])
            for partition in partitions:
                I_internal.update(self.findIndependenceWithinPartition(L, partition))
                Vs.update(partition)

            I.update(I_internal)
            E_temp = self.getEdgesGivenIndependence(Vs, I_internal)
            E.update(E_temp)

        LOGGER.info("Discovered edges:")
        for edge in E.values():
            arrow = "-->" if edge[2] else "---"
            LOGGER.info(f"     {edge[0]} {arrow} {edge[1]}")

        # Finally, prune edges and perform collider tests
        E = self.pruneEdges(I, E)
        LOGGER.info("After pruning, edges are:")
        for edge in E.values():
            arrow = "-->" if edge[2] else "---"
            LOGGER.info(f"     {edge[0]} {arrow} {edge[1]}")

        # Perform collider tests
        O, E = self.colliderTest(E, I)
        return I, O, E

    def findIndependenceForNonAtomic(self, L: Cover):
        """
        Use CrossCoverTest to check for independence relationships amongst a
        non atomic Cover L and its children.

        Returns:
            I: independence relationships
        """
        assert not L.isAtomic, f"{L} should be non atomic."
        I = Independences()
        children = self.G.latentDict[L]["children"]
        subcovers = self.G.latentDict[L]["subcovers"]

        # Independence: children indep all other variables | L
        adjacentNodes = set()
        for subcover in subcovers:
            adjacentNodes.update(self.G.findAdjacentNodes(subcover))
        adjacentNodes -= subcovers
        for N in adjacentNodes:
            for C in children:
                I[frozenset([N, C])] = {
                    "condition": frozenset(subcovers),
                    "setA": set(),
                    "setB": set(),
                    "k": len(L),
                    "G": None,
                }

        # Independence: Check amongst children and L
        for C in children:
            Gp = deepcopy(self.G)
            Gp.makeRoot(C)
            Gp = Gp.disconnectForNonAtomicParents(Gp, L)

            # Test every pair to find potential CI relationship between them
            Vs = Gp.latentDict[L]["subcovers"] | set([C])
            I.update(self._crossCoverTest(Gp, Vs))

        # Independence: If A, B form L, then check for A indep B | parents(A, B)
        pairs = list(combinations(subcovers, 2))
        for pair in pairs:
            Gp = deepcopy(self.G)
            A, B = pair

            # Either A or B can be root, but not both, hence we try each in turn
            for V in [A, B]:
                parentsV = Gp.findParents(V)
                if len(parentsV) > 0:
                    Gp.disconnectNodes(parentsV, set([V]))
                    for P in parentsV:
                        Gp.makeRoot(P)
                    break

            # Now the other variable cannot be root, and we do likewise to it
            W = B if (A == V) else A
            parentsW = Gp.findParents(W)
            Gp.disconnectNodes(parentsW, set([W]))
            for P in parentsW:
                Gp.makeRoot(P)

            # Get parents(A, B) and run crossCoverTest
            atomicParents = set()
            for P in parentsV | parentsW:
                if not P.isAtomic:
                    atomicParents.update(Gp.latentDict[P]["subcovers"])
                else:
                    atomicParents.add(P)
            atomicParents -= set([A, B])
            I.update(self._crossCoverTest(Gp, set([A, B]), atomicParents))

        for k, v in I.items():
            A, B = list(k)
            LOGGER.info(f"   Found {A} indep {B} | {v['condition']}..")
        return I

    def findIndependenceAcrossPartitions(self, L: Cover):
        """
        Given a locality L, test for all CI relationships conditional on L
        amongst all of L's adjacent nodes (N).
        Args:
            L: The locality L that we are testing
            As: The adjacent nodes to L that form the test set
            Bs: The adjacent nodes to L that form the control set

        Returns:
            I: set of conditional independences
            partitions: partitions of N that are CI on L from each other
        """
        # Orient Graph at L
        Gp = deepcopy(self.G)
        Gp.makeRoot(L)
        children = Gp.findChildren(L)
        nonAtomicParents = Gp.findParents(L, non_atomic=True)
        assert (
            len(nonAtomicParents) <= 1
        ), f"{L} should not have more than one non-atomic parent."

        # Make Test and Control set
        As = set()
        for child in children:
            if child.isTemp:
                As.update(Gp.findChildren(child))
            else:
                As.add(child)
        if len(nonAtomicParents) == 1:
            P = next(iter(nonAtomicParents))
            Gp = Gp.disconnectForNonAtomicParents(Gp, P)
            for subcover in Gp.latentDict[P]["subcovers"]:
                As.add(subcover)
        Bs = Gp.findNonAtomicChildren(L)  # control set

        I = self._findIndependences(Gp, L, As, Bs)

        if len(I) == 0:
            # No independences found, assume that children indep others | L
            I = self._assumeFullIndependences(L, As, Bs)

        partitions = self._findPartitions(I, L, As | Bs)

        LOGGER.info(f"Found independences at {L}..")
        return I, partitions

    def _assumeFullIndependences(
        self, L: Cover, testSet: set[Cover], controlSet: set[Cover]
    ):
        """
        Instead of testing for independence, we assume that all minimal subsets
        of the variables in testSet are conditionally independent of all other
        variables after conditioning on L.
        """
        I = Independences()
        allSubsets = M.generateSubsetMinimal(testSet, len(L))
        for As in allSubsets:
            Bs = setDifference(testSet, As) | controlSet  # control set
            for A in As:
                for B in Bs:
                    I[frozenset([A, B])] = {
                        "condition": frozenset([L]),
                        "setA": set(),
                        "setB": set(),
                        "k": len(L),
                        "G": None,
                    }
        for k in I:
            A, B = list(k)
            LOGGER.info(f"   Assume {A} indep {B} | {L}..")
        return I

    def _findIndependences(
        self, G: LatentGroups, L: Cover, testSet: set[Cover], controlSet: set[Cover]
    ):
        if setLength(testSet | controlSet) < 2 * len(L) + 2:
            LOGGER.info(f"Not enough adjacent nodes at {L}..")
        Gp = deepcopy(G)

        I = Independences()
        for As in M.powerset(testSet):
            As = set(As)  # test set
            Bs = setDifference(testSet, As) | controlSet  # control set

            # Prune Bs if there are backdoor connections from As
            Bs = Gp.pruneControlSet(Gp, As, Bs)

            # Then disconnect all edges to L for testing
            Gp = Gp.disconnectAllEdgestoCover(Gp, L)

            # Check if test/control set have enough cardinality for test
            if (setLength(As) <= len(L)) or (setLength(Bs) <= len(L)):
                continue

            # As must not contain more than k elements from
            # any atomic Cover with cardinality <= k-1
            if Gp.containsCluster(As):
                continue

            test, rk = self.structuralRankTest(Gp, As, Bs, len(L))
            if test and rk == len(L):
                for A in As:
                    for B in Bs:
                        I[frozenset([A, B])] = {
                            "condition": frozenset([L]),
                            "setA": set(),
                            "setB": set(),
                            "k": rk,
                            "G": None,
                        }
        for k in I:
            A, B = list(k)
            LOGGER.info(f"   Found {A} indep {B} | {L}..")
        return I

    def _findPartitions(self, I: dict, L: Cover, Ns: set[Cover]):
        """
        Given independences and the set of nodes, partition the nodes into
        clusters such that there are no edges between clusters apart from
        edges through L.
        Args:
            I: Dict of independence relationships
            L: The locality L that we are testing
            Ns: All adjacent nodes to L

        Returns:
            partitions: partitions of Ns that are CI on L from each other
        """
        edges = {N: set() for N in Ns}
        for N1 in Ns:
            for N2 in Ns:
                if (N1 != N2) and (frozenset([N1, N2]) not in I):
                    edges[N1].add(N2)
                    edges[N2].add(N1)

        partitions = []
        nodes = deepcopy(Ns)
        while len(nodes) > 0:
            partition = set()
            Q = deque()
            L = nodes.pop()
            Q.append(L)
            while len(Q) > 0:
                L = Q.pop()
                partition.add(L)
                for neighbour in edges[L]:
                    if neighbour not in partition:
                        Q.append(neighbour)
            nodes = nodes - partition
            partitions.append(partition)
        LOGGER.info(f"Found the following partitions:")
        for partition in partitions:
            LOGGER.info(f"    {','.join([str(v) for v in partition])}")
        return partitions

    def findIndependenceWithinPartition(self, L: Cover, Ns: set[Cover]):
        """
        Use CrossCoverTest to check for independence relationships amongst L
        and variables Ns within a partition.

        Returns:
            I: independence relationships
        """
        I = Independences()

        # Orient Graph at L
        Gp = deepcopy(self.G)
        Gp.makeRoot(L)

        # Represent non-atomic parents of L correctly
        nonAtomicParents = Gp.findParents(L, non_atomic=True)
        if len(nonAtomicParents) == 1:
            P = next(iter(nonAtomicParents))
            Gp = Gp.disconnectForNonAtomicParents(Gp, P)

        # Test every pair to find potential CI relationship between them
        Vs = Ns | set([L])
        I.update(self._crossCoverTest(Gp, Vs))
        return I

    def getEdgesGivenIndependence(self, Ns: set[Cover], I: dict):
        """
        Given a set of nodes Ns and independence relationships between them,
        return the set of edges that should exist between them.

        1. Measures cannot have relationship with another measure
        2. Relationship between latent and measure is always latent -> measure

        Returns:
            E: A dict where the key is a frozenset([A, B]) of variables and the
               value is a triplet (A, B, 0 or 1) where 0 means undirected and
               1 means directed.
        """
        E = Edges()
        pairs = list(combinations(Ns, 2))
        for pair in pairs:
            pair = frozenset(pair)
            if not pair in I.keys():
                A, B = list(pair)
                key = frozenset([A, B])
                if (not A.isLatent) and (not B.isLatent):
                    continue
                elif (not A.isLatent) and (B.isLatent):
                    E[key] = (B, A, 1)
                elif (A.isLatent) and (not B.isLatent):
                    E[key] = (A, B, 1)
                else:
                    E[key] = (A, B, 0)
        LOGGER.info("Edges are:")
        for edge in E.values():
            arrow = "-->" if edge[2] else "---"
            LOGGER.info(f"     {edge[0]} {arrow} {edge[1]}")
        return E

    def pruneEdges(self, I: Independences, E: Edges):
        """
        Given independence relationships I and a candidate dict of edges E, we
        further prune edges in E that are not compatible with I.

        Returns:
            E: pruned dict of edges
        """
        pruned = Edges()
        for edge in E:
            if edge in I:
                continue
            pruned[edge] = E[edge]
        return pruned

    def _crossCoverTest(self, G, Vs, Cs=None):
        """
        Internal function for crossCoverTest
        - Vs: The set of variables to test if they are CI from each other
        - Cs (optional): The set of variable to condition on. If None, then
                         the unused variables in Vs will be used to condition.
        """
        independences = {}
        # Find all combinatorial pairs
        pairs = list(itertools.combinations(Vs, 2))
        Ls = set([L for L in Vs if L.isLatent])
        for pair in pairs:
            pair = set(pair)

            # Create powerset of variables to condition on
            if Cs is None:
                Vsets = list(M.powerset(Ls - pair))
            else:
                Vsets = list(M.powerset(Cs))

            A = pair.pop()
            B = pair.pop()
            for Vset in Vsets:

                # Create dictionary of each V in Vset to their children
                Vdict = {}
                for V in Vset:
                    Vdict[V] = G.findChildren(V)

                tester = CrossCoverTester(Vdict, A, B)
                setA, setB, cardinality, satisfied = tester.search()

                # Test if A indep B | Vset
                if satisfied:
                    test, rk = self.structuralRankTest(
                        G, setA, setB, k=tester.getExpectedRank()
                    )
                    if test:
                        key = frozenset([A, B])  # A indep B | Vset
                        independences[key] = {
                            "condition": frozenset(Vset),
                            "setA": setA,
                            "setB": setB,
                            "k": rk,
                            "G": deepcopy(G),
                        }
        return independences

    def colliderTest(self, E: dict, I: dict):
        """
        Find all possible colliders amongst the edges in E and using information
        from conditional independences found in I.

        Args:
            E: A dict of edges (conformant with I)
            I: A dict of conditional independence relationships

        Returns:
            O: A set of collider triplets (A, B, D) s.t. A -> D <- B
        """
        # Build dictionary of edges so that we can query for adjacent nodes
        edgeDict = {}
        for pair in E:
            A, B = pair
            edgeDict[A] = edgeDict.get(A, set()) | set([B])
            edgeDict[B] = edgeDict.get(B, set()) | set([A])

        O = set()
        for AB, details in I.items():
            if I[AB]["setA"] == set():
                continue

            # For each indep, we have A indep B | Cs
            A, B = list(AB)
            Cs = details["condition"]

            # Find potential colliders Ds s.t. A - D - B:
            Ds = edgeDict[A].intersection(edgeDict[B])
            Ds -= set([A, B]) | Cs

            O.update(self._colliderTest(A, B, details, Ds))

        # Use colliderSet to add directions to edges in edgeSet
        for collider in O:
            A, B, D = collider
            LOGGER.info(f"   Found collider: {A} -> {D} <- {B}")
            AD = frozenset([A, D])
            BD = frozenset([B, D])
            assert AD in E, "AD should be an edge."
            assert BD in E, "BD should be an edge."
            E[AD] = (A, D, 1)
            E[BD] = (B, D, 1)
        return O, E

    def _colliderTest(self, A: Cover, B: Cover, details: dict, Ds: set[Cover]):
        """
        Test if there are colliders at A->D<-B and return set of colliders.

        Args:
            A, B: Variables that we are testing
            details: details from the independence test
            Ds: variables that are potential colliders
        """
        O = set()
        setA = details["setA"]
        setB = details["setB"]
        k = details["k"]
        Gp = details["G"]  # The state of the graph when indep was found
        for D in Ds:
            setA1 = setA | set([D])
            setB1 = setB
            test1, rk1 = self.structuralRankTest(Gp, setA1, setB1, k=k)

            setA2 = setA
            setB2 = setB | set([D])
            test2, rk2 = self.structuralRankTest(Gp, setA2, setB2, k=k)

            # Check if rk1 > k < rk2
            if (not test1) and (not test2):
                O.add((A, B, D))  # A -> D <- B
        return O

    # Sample of generated data from g
    def addSample(self, df):
        self.df = df
        self.rankTester = CCARankTester(df, alpha=self.alpha)

    def sampleRankTest(self, A, B, rk):
        """
        Test null hypothesis that rank(subcovariance[A,B]) <= rk
        A, B: Columns for testing
        """
        if rk == 0:
            return False
        cols = list(self.df.columns)
        I = [cols.index(a) for a in A]
        J = [cols.index(b) for b in B]
        test = self.rankTester.test(I, J, r=rk)
        return not test

    def sampleRankP(self, A, B, rk):
        cols = list(self.df.columns)
        I = [cols.index(a) for a in A]
        J = [cols.index(b) for b in B]
        p = self.rankTester.prob(I, J, r=rk)
        return p

    # From a set of measured variables, extract the variable names
    def getVarNames(self, As):
        measuredVars = []
        for A in As:
            if A.isLatent:
                assert False, "A is not a measured var set"
            if len(A) > 1:
                assert False, "Measured Vars should not have more than 1"
            measuredVars.append(A.takeOne())
        return measuredVars

    def structuralRankTest(self, G, As, Bs, k):
        """
        Test if As forms a cluster by seeing if rank(subcov[A,B]) <= k.

        Returns tuple of whether rank is deficient and lowest rank tested.
        """
        Ameasures = G.pickAllMeasures(As)
        Ameasures = self.getVarNames(Ameasures)
        Bmeasures = G.pickAllMeasures(Bs)
        Bmeasures = self.getVarNames(Bmeasures)

        if self.sample:
            test = self.sampleRankTest(Ameasures, Bmeasures, k)
        else:
            test = self.g.rankTest(Ameasures, Bmeasures, k)

        if not test:
            return (False, None)

        # If rank is deficient, keep testing until lowest rank
        rk = k
        if test and k > 1:
            test2, rk2 = self.structuralRankTest(G, As, Bs, k - 1)
            if test2:
                test = test2
                rk = rk2
        return (test, rk)
