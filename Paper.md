# Interplay of coil–globule transitions and aggregation in homopolymer aqueous solutions: Simulation and topological insights

Cite as: J. Chem. Phys. 163, 191101 (2025); doi: 10.1063/5.0280838 Submitted: 14 May 2025 • Accepted: 31 October 2025 • Published Online: 20 November 2025 

Junichi Komatsu,<sup>1</sup> Kenichiro Koga,<sup>1,2,a)</sup> and Jonas Berx<sup>3,a)</sup> 

## AFFILIATIONS

<sup>1</sup> Department of Chemistry, Faculty of Science, Okayama University, 3-1-1 Tsushima-Naka, Kita-ku, Okayama 700-8530, Japan <sup>2</sup>Research Institute for Interdisciplinary Science, Okayama University, 3-1-1 Tsushima-Naka, Kita-ku, Okayama 700-8530, Japan <sup>3</sup>Niels Bohr International Academy, Niels Bohr Institute, University of Copenhagen, Blegdamsvej 17, 2100 Copenhagen, Denmark 

Note: This paper is part of the Special Topic, Carlos Vega Festschrift. <sup>a)</sup>Authors to whom correspondence should be addressed: koga@okayama-u.ac.jp and jonas.berx@nbi.ku.dk 

## ABSTRACT

We investigate the structural and topological properties of hydrophobic homopolymer chains in aqueous solutions using molecular dynamics simulations and circuit topology (CT) analysis. By combining geometric observables, such as the radius of gyration and the degree of aggre gation, with CT data, we capture the relationship between coil–globule and aggregation transitions, resolving the system’s structural changes with temperature. Our results reveal a temperature-driven collective transition from isolated coiled chains to globular aggregates. At a char acteristic transition temperature T , each chain in multichain systems undergoes a rapid coil–globule collapse, coinciding with aggregation, in contrast to the gradual collapse observed in single-chain systems at infinite dilution. This collective transition is reflected in geometric descriptors and a reorganization of CT motifs, shifting from intrachain-dominated motifs at low temperatures to a diverse ensemble of multi chain motifs at higher temperatures. CT motif enumeration provides contact statistics while offering a topologically detailed view of polymer organization. These findings highlight CT’s utility as a structural descriptor for polymer systems and suggest applications for biopolymer aggregation and folding. 

© 2025 Author(s). All article content, except where otherwise noted, is licensed under a Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC) license (https://creativecommons.org/licenses/by-nc/4.0/). https://doi.org/10.1063/5.0280838 

The coil–globule transition in homopolymer solutions is a conformational change from an extended coil to a compact globular state, driven in many cases by varying temperature.<sup>1–5</sup> It may also be induced by changing cosolvent concentration<sup>6,7</sup> or adding specific salts.<sup>8,9</sup> While this transition is ordinarily gradual in dilute systems, its sharpness can vary depending on the polymer type and solution conditions.<sup>3,4</sup> 

In an organic solvent, linear flexible homopolymer chains adopt a coil state at high temperatures and a globule state at low temperatures around the upper critical solution temperature.<sup>1–3</sup> The primary driving force for polymer chain collapse is typically van der Waals intrachain interactions, which are weak but become more significant at lower temperatures. 

In contrast, in aqueous solutions, water-soluble polymer chains are extended at low temperatures and globular at high temperatures.<sup>4,5,10</sup> The opposite temperature dependence indicates that the driving force is the hydrophobic interaction between nonpolar moieties, an effective force that is weakly attractive or could even be repulsive at low temperatures, but is strongly attractive at elevated temperatures. For example, the osmotic second virial coefficient of methane in water is a decreasing function of temperature, changing from a positive value near 273 K to large negative values at higher temperatures.<sup>11</sup> The water-mediated hydrophobic interactions are largely entropic in origin: the configurational entropy of water is greater when nonpolar solutes (moieties) are in contact with each other than when they are apart.<sup>12</sup> Such interactions play a key role in determining the native structures of biological macromolecules and in driving self-assembly into ordered forms like micelles, vesicles, or lamellar phases.<sup>13</sup> 

Significant effort has been devoted to studying the nature of the coil-to-globule transition of a single polymer chain, including the fully collapsed globule state, using highly dilute solutions with narrow molecular weight distributions.<sup>3–5,10</sup> At higher polymer concentrations, chain aggregation occurs at a specific temperature, making it difficult to isolate information about the single-chain coilto-globule transition from experiments. Previous simulations and finite-size scaling studies of semidilute polymer solutions show that single-chain collapse and multi-chain aggregation occur at the same temperature for infinitely long chains and that the upper consolute point for finite-length chains belongs to the three-dimensional Ising universality class.<sup>14,15</sup> In contrast, real systems, such as cellular aqueous environments and soft matter materials, have, in general, high polymer concentrations. Under these conditions, the intrachain coilto-globule transition often occurs alongside interchain aggregation within a specific temperature range. This study aims to investigate the coupling between the intrachain coil-to-globule transition and interchain aggregation. 

Experimental studies of the relationship between chain collapse and aggregation have been conducted under non-equilibrium conditions, specifically by quenching poly(methyl methacrylate) (PMMA) in solvents into the phase-separated regime.<sup>16–18</sup> At low concentrations, individual chains first collapse, followed by the aggregation of the collapsed chains into clusters of varying sizes. Within a specific concentration range, a sequential transition from chain collapse to chain aggregation is observed, with the extent of overlap between the two processes diminishing as concentration decreases. A prototypical example of polymers in aqueous solutions undergoing coil-to-globule phase transitions upon heating is poly(N-isopropylacrylamide) (PNIPAM), which exhibits this transition around 305 K $( 3 2 ^ { \circ } \mathrm { C } ) . ^ { 4 , 1 9 }$ Depending on the concentration, PNIPAM can undergo either distinct or collective coil-to-globule and aggregation transitions.<sup>20</sup> 

We study a model system of homopolymer aqueous solutions using molecular dynamics (MD) simulation and topological analysis. The model polymer solution is designed such that, at infinite dilution, a single polymer chain undergoes a gradual coil–globule transition around room temperature.<sup>21</sup> Here, we investigate the structural behavior of multiple homopolymer chains at finite concentration, focusing on the interplay between the coil–globule transition and their aggregation in water. 

These conformational and collective transitions naturally raise important questions about the underlying topology of polymer chains and their interactions. Understanding the structural organization of multi-chain systems requires a framework capable of describing not only spatial configurations but also the connectivity and entanglement of chains. 

The two main approaches to topologically describe a system of linear chains, such as proteins, peptides, or RNA molecules, are knot theory<sup>22,23</sup> and circuit topology (CT).<sup>24–26</sup> Both frameworks decompose a set of entangled polymers into fundamental units. For CT, these fundamental units are topological “motifs” formed by the mutual relation between two contact pairs, while knot theory decomposes the system into combinations of prime knots. Circuit topology serves as a complementary approach to knot theory since it is able to describe hard contacts (i.e., chemical or physical bonds between atoms, residues, etc.) for open chains, while knot theory focuses on entanglement and requires the chains to be closed. While CT can be extended to describe single-chain entanglement (soft contacts),<sup>27,28</sup> we will not focus on that aspect here. Since our system consists of open chains, we will, therefore, analyze it within the CT framework. Since CT mainly concerns the structures formed by interacting chains, it is ideally suited to study structural phase transitions.<sup>29</sup> 

Let us set the stage for the polymer model studied in this study. We simulate m linear freely linked chains of n spherical hydrophobic monomers with fixed intermonomer distances of $b = 0 . 3 4 5$ nm in the TIP4P/2005 water model.<sup>30</sup> The simulated systems comprise either $m = 4$ or $m = 8$ polymer chains with $n = 3 0$ monomers each, along with 8000 water molecules. For $m = 4 ,$ , the representative number density of monomers (at 298 K and 1 bar) is $\stackrel { \cdot } { 0 . 4 9 } \mathrm { n m } ^ { - 3 }$ , and the corresponding concentration is 0.81 mol/l; for $m = 8 ,$ these values are doubled. 

The potential energy of the system is the sum of monomer–monomer, monomer–water, and water–water pair potentials. The first two pair potentials are Lennard-Jones (LJ) potentials, 

$$
\phi_ {\mathrm{LJ}} (r) = 4 \epsilon \left[ \left(\frac {\sigma}{r}\right) ^ {1 2} - \left(\frac {\sigma}{r}\right) ^ {6} \right],\tag{1}
$$

while the water–water pair potential is a sum of the LJ potential for the pair of oxygen sites and Coulomb potentials for the charged sites, as described by the TIP4P/2005 model. The cutoff distance for the LJ potential was set to 1.1 nm, while the Coulomb potential was evaluated by the particle mesh Ewald method with the real space cutoff distance of 1.1 nm. 

The LJ parameters, $\epsilon _ { \mathrm { m } }$ and $\sigma _ { \mathrm { m } } .$ , for the monomer–monomer interactions are those for methane in the TraPPE-UA model:<sup>31</sup> $\epsilon _ { \mathrm { m } } = 1 . 2 3 0$ kJ $\mathrm { m o l ^ { - 1 } }$ and $\sigma _ { \mathrm { m } } = 0 . 3 7 3 0$ nm. The LJ parameters $\epsilon _ { \mathrm { w m } }$ and $\sigma _ { \mathrm { w m } }$ for the oxygen (of $_ \mathrm { H _ { 2 } O ) }$ -monomer pair interaction are $\epsilon _ { \mathrm { w m } } = 1 . 3 5 6$ kJ mol<sup>−1</sup> and $\sigma _ { \mathrm { w m } } = 0$ .3444 nm. With these LJ parameters, the simple model polymer chain in the TIP4P/2005 water undergoes the coil–globule transition near room temperature.<sup>21</sup> We confirmed that for both monomer concentrations of 0.81 and 1.62 mol $\mathrm { ~ L ~ } ^ { - 1 }$ , water and individual monomers mix at all temperatures from 240 to 360 K. 

MD simulations were performed using GROMACS $2 0 1 8 ^ { 3 2 }$ in the isothermal–isobaric (NpT) ensemble, using periodic boundary conditions and a time step of 1 fs. The coordinates were sampled every 50 fs. The pressure was maintained at 1 bar using the Parrinello–Rahman method, and the temperature was controlled using the Nosé–Hoover method. The simulation time was 50 ns after equilibration at each temperature. Representative configurations for a system with four chains as a function of temperature are shown in Fig. 1. 

Together with topological measures, we use two geometric ones: the radius of gyration, $R _ { \mathrm { g } } ,$ a measure of the compactness of a polymer chain, and the degree of aggregation, $D _ { \mathrm { c } } ,$ a measure of the inhomogeneity of the polymer solution. For a polymer with n monomers at coordinates r<sub>i</sub>, 

$$
R _ {g} = \sqrt {\frac {1}{n} \left\langle \sum_ {i = 1} ^ {n} \left(\mathbf {r} _ {i} - \mathbf {r} _ {c}\right) ^ {2} \right\rangle},\tag{2}
$$

where $\langle \cdots \rangle$ denotes the ensemble average over all polymers and the time average. The progress of the coil–globule transition is measured by the temperature dependence of $R _ { \mathrm { g } } .$ . Similarly, the degree of aggregation of polymers is evaluated by 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-09/788fa88a-11a4-4082-b023-d6bf5fb6be7b/f42d41c5c51aec23ed76d52b47e37e63faa0c813f613530889da55e810dc031f.jpg)



FIG. 1. System configurations for different temperatures, showing the structural transition at a characteristic temperature $T _ { c } \approx 2 9 0$ , for m = 4 polymers with $ { n } = 3 0$ monomers each. The polymers collapse from a dilute coiled state into a globular aggregated state.


$$
D _ {c} = \left\langle \left| \mathbf {r} _ {c} ^ {\alpha} - \mathbf {r} _ {c} ^ {\beta} \right| \right\rangle ,\tag{3}
$$

where α and $\beta$ are indices assigned to the polymers, and $\langle \cdots \rangle$ now represents the average over all polymer pairs in water and over time. We note that $D _ { \mathrm { c } }$ for an ideal-gas model is proportional to $L ,$ the side length of the simulation box under periodic boundary conditions. Therefore, we will present the dimensionless quantity $D _ { \mathrm { c } } / L$ in Fig. 5(b). 

For comparison purposes, in addition to the model polymer solution, we examine the forced-coil systems and the forced-globule system. For the former, we fix the polymer configuration by imposing constraints on monomer distances and angles. Specifically, the inter-monomer distance is set to 2.949 nm for monomers numbered 1–11 and 20–30 and to 2.660 nm for monomers numbered 6–15, 11–20, and 16–25. The angle formed by three points is set to $1 7 6 . 1 4 ^ { \circ }$ for the triplets 1–11–20 and 11–20–30. Note that the numbering runs from one end of the chain to the other. For the forced-globule system, we increase the Lennard-Jones (LJ) energy parameter for monomer–monomer intrachain interactions from $\epsilon _ { \mathrm { m } }$ to $1 0 \epsilon _ { \mathrm { m } } ,$ while keeping the parameter for interchain interactions unchanged, collapsing the polymer into a globular state. 

For the circuit topological description of the system, let us first consider the arrangement of contacts on a single linear chain. This arrangement is a topological property, invariant to chain folding or stretching. In circuit topology (CT), motifs are defined as pairwise topological arrangements of contacts, e.g., contacts α and β. 

Contacts α and β may involve $\mathcal { M } = 1$ to 4 chains. CT provides a second-order topological description of the system. In the firstorder description, considering individual contacts only, CT motifs are undefined, and contacts are classified as intrachain (within one chain) or interchain (between two chains). This analysis extends to tertiary, quaternary, or higher-order arrangements. As the number of distinct topological motifs grows exponentially with the number of contacts, we limit our analysis to binary arrangements of contacts, consistent with current practice. 

Each contact involves two monomer sites, and each motif, a topological arrangement of two contacts α and $\beta ,$ involves four monomers. For $\bar { \mathcal { M } } = 1 .$ , three motifs are possible: series (S), parallel (P), and cross (X). These are visualized in Fig. 2. The motifs describe the topological relation between two loops in a single chain, classified as topologically independent (S) or bound (P and X). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-09/788fa88a-11a4-4082-b023-d6bf5fb6be7b/4f6ff49a734cbd7ace3f901e173919e90467c665798ba93a46d8a4a8cda60543.jpg)



FIG. 2. Table of single-chain circuit topology motifs. Contacts α, β are represented by green and purple lines connecting contact sites (white circles). Underneath each motif, a cartoon representation is shown illustrating the concomitant topology.


In a multichain polymer system, additional motifs are identified by allowing contacts between distinct chains.<sup>29,33</sup> Since four sites of two contacts can be distributed among different chains, the maximum number of chains in a single CT motif is $\mathcal { M } _ { \mathrm { m a x } } = 4$ . All multichain CT motifs are illustrated in Fig. 3, along with a simplified cartoon representation. 

For $\mathcal { M } = 2 ,$ three motifs are identified. The tandem (T ) motif is formed by one intrachain and one interchain contact; the loop (L<sub>2</sub>) motif by two interchain contacts; and the independent (I<sub>2</sub>) motif by two intrachain contacts, each on its respective polymer. Note that I<sub>2</sub> is only determined topologically. The two chains can be geometrically distant or close, provided each has an intrachain contact. 

For $\mathcal { M } = 3 .$ , two motifs are identified: the tandem (T ) motif, comprising two interchain contacts involving one common chain and two others, and the independent (I<sub>3</sub>) motif, comprising one intrachain contact on one chain and one interchain contact between two others. Finally, for $\mathcal { M } = 4 ,$ only one motif exists: the independent $\left( \mathrm { I } _ { 4 } \right)$ motif, with two interchain contacts, each between a distinct pair of chains. 

The system’s first-order description, i.e., the number of intrachain and interchain contacts, follows from the exact counts of CT motifs. Let $N _ { \mathrm { i n t r a } }$ and $N _ { \mathrm { i n t e r } }$ denote the number of intrachain and interchain contacts, respectively, in the system. The total number of contacts is $N = N _ { \mathrm { i n t r a } } + N _ { \mathrm { i n t e r } }$ . The CT motif counts are denoted S, P, X, T<sub>2</sub>, L<sub>2</sub>, I<sub>2</sub>, $T _ { 3 } , I _ { 3 } ,$ , and $I _ { 4 } .$ . Their fractions are denoted by corresponding lowercases: s, $p , x , t _ { 2 } , l _ { 2 } , i _ { 2 } , t _ { 3 } , i _ { 3 } .$ , and i<sub>4</sub>. 

When there are N contacts in a multichain polymer system, a given intrachain contact generates $N - 1$ motifs involving at least one intrachain contact, i.e., S, P, X, I , $\mathrm { T } _ { 2 } ,$ and I . As there are $N _ { \mathrm { i n t r a } }$ intrachain contacts, $( N - 1 ) N _ { \mathrm { i n t r a } }$ would be the total number of such motifs. However, the motifs S, P, X, and $\mathrm { I } _ { 2 }$ involve two intrachain contacts and, therefore, they are doubly counted in $( N - 1 ) N _ { \mathrm { i n t r a } }$ Consequently, $\bigl ( N - 1 \bigr ) N _ { \mathrm { i n t r a } } = 2 \bigl ( \dot { S ^ { } } + P + X + I _ { 2 } \bigr )$ $+ \ T _ { 2 } + I _ { 3 }$ or, equivalently, 

$$
N _ {\text { intra }} = \frac {1}{N - 1} \left[ 2 (S + P + X + I _ {2}) + T _ {2} + I _ {3} \right].\tag{4}
$$

The fraction of intrachain contacts $r _ { \mathrm { i n t r a } } = N _ { \mathrm { i n t r a } } / N$ is then 

$$
\begin{array}{r l} r _ {\text { intra }} & = \frac {1}{N (N - 1)} \left[ 2 (S + P + X + I _ {2}) + T _ {2} + I _ {3} \right] \\ & = \frac {1}{M} (S + P + X + I _ {2}) + \frac {1}{2 M} (T _ {2} + I _ {3}) \\ & = (s + p + x + i _ {2}) + \frac {1}{2} (t _ {2} + i _ {3}), \end{array}\tag{5}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-09/788fa88a-11a4-4082-b023-d6bf5fb6be7b/21d03c6530cdc0fb4ca9416477e586122433b63870a4fab42fbe7eff3355f6f5.jpg)



FIG. 3. Table of single-chain circuit topology motifs. Contacts α, β are represented by green and purple lines connecting contact sites (white circles). Next to each motif, a cartoon representation is shown illustrating the concomitant topology. Note that when a motif is degenerate, only one example is shown.


where in the second line we have used the fact that the total number of CT motifs is $M = N ( N - 1 ) / 2$ . Similarly, the fraction of interchain contacts is given by 

$$
r _ {\text { inter }} = \left(l _ {2} + t _ {3} + i _ {4}\right) + \frac {1}{2} \left(t _ {2} + i _ {3}\right).\tag{6}
$$

It is thus straightforward to see that CT provides a more detailed level of description than simply classifying contacts as either intra- or interchain, but the latter can be recovered through simple combinatorics. The number and relative fraction of contacts of both types change drastically as a function of temperature. In Fig. 4, we show the exact number of contacts and contact fractions as a function of temperature T and compare the results with those obtained through the enumeration of CT motifs according to Eqs. (5) and (6) (black symbols). The results coincide exactly. 

We now present our numerical results and examine single-polymer conformation, multichain configuration, and their relationship by tracking the radius of gyration $\left( \mathrm { { \bar { { R } } _ { g } } } \right)$ , interchain distance (D<sub>c</sub>), and CT motif fractions as a function of temperature (240–360 K), as shown in Figs. 5 and $^ { 6 , }$ respectively. 

Figure 5(a) shows that, for both systems with m = 4 and 8 polymer chains, $R _ { \mathrm { g } }$ gradually decreases with increasing temperature T from 240 to 280 K and drops sharply at 290 K. Therefore, the structural change of each polymer at 290 K may be termed the coil-to-globule transition. Below this temperature, the polymers are extended and well-solvated, which is consistent with the fact that the osmotic second virial coefficient for hydrophobic molecules in water is positive at low temperatures.<sup>11</sup> The system exhibits mostly the single-chain motifs S, P, and X, and ${ \mathrm { I } } _ { 2 } ,$ all of which increase their counts gradually as T increases from 240 to 280 K, as shown in Fig. 6(a). The system is enriched in I<sub>2</sub> and S, indicating that loops are formed independently; if two loops are formed on the same chain, they are far enough apart such that they do not interact. This is a distinct property of the coiled state of the polymers: they are stretched out spatially and form loops only locally, leading to a large radius of gyration. The gradual increase in single-chain motifs indicates that, as T increases from 240 to 280 K, more loops form, which can interact by looping back onto themselves, sometimes forming the X motif. Since the contraction of a polymer is accompanied by loop formation, $R _ { \mathrm { g } }$ decreases; it does so slowly because the conformational change of each polymer chain occurs independently in this temperature range. Figure 4(a) also shows that $N _ { \mathrm { i n t r a } }$ for the multichain system gradually increases up to 280 K, just as does $N _ { \mathrm { i n t r a } }$ for the single-chain system. 

At a characteristic temperature T<sub>c</sub> ≈ 290 K, however, $N _ { \mathrm { i n t r a } }$ sharply increases [Figs. 4(a)-4(c)], $R _ { \mathrm { g } }$ sharply decreases [Fig. 5(a)], and the number of single-chain motifs increases drastically. These indicate that the polymers collectively undergo a coil-to-globule transition at $T _ { \mathrm { c } }$ as they aggregate due to hydrophobic interactions. Conversely, $R _ { \mathrm { g } }$ and $N _ { \mathrm { i n t r a } }$ for the single-chain system exhibit no discontinuous behavior over the whole range. In essence, interpolymer effective interactions in water, $\mathrm { i . e . , }$ the hydrophobic interactions, sharpen an otherwise gradual coil-to-globule conformational change. For comparison, the $R _ { \mathrm { g } }$ values for two model systems containing either four forced-coiled or forced-globular polymers are also plotted in Fig. 5(a). 

Figure 5(b) shows how $D _ { \mathrm { c } } / L$ varies with temperature for the four model systems. The $D _ { c } / L$ values for the standard and forcedglobule systems exhibit a sharp decrease as $T$ increases from 280 to 290 K, indicating that aggregation occurs sharply at approximately T<sub>c</sub> ≈ 290 K. We also note that the aggregation temperatures for the two systems with m = 4 and 8 coincide, indicating that $T _ { \mathrm { c } }$ is only weakly dependent on the monomer concentration in the examined range. Above $T _ { \mathrm { c } } ,$ the polymers have a reduced solventaccessible surface area due to both the coil-to-globule transition and aggregation. This is readily observed in the CT spectrum (Figs. 4 and 6): for $T < T _ { \mathrm { c } }$ , no motifs exhibiting interchain interactions can exist $( N _ { \mathrm { i n t e r } } = 0 )$ . At $T _ { \mathrm { c } } , \ N _ { \mathrm { i n t e r } }$ sharply jumps to a non-zero value, and $N _ { \mathrm { i n t r a } }$ also rises significantly, suggesting a collective collapse-aggregation transition. With increasing temperature, the total number of contacts—and, therefore, the CT motifs—decreases, but their relative fractions remain stationary. This indicates that the system reaches a stable macroscopic configuration where the topology remains similar, but thermal fluctuations can disrupt existing contacts, allowing individual chains to rearrange in confined environments. In addition, the hydrophobic effect, manifested as low solubility and hydrophobic monomer–monomer attraction, may be maximal around $T _ { \mathrm { c } }$ . Beyond this point, more water molecules may penetrate the polymer aggregate, thereby reducing intra- and inter-monomer contacts. For $\dot { T } > T _ { c } ,$ the motifs I<sub>3</sub> and $\mathrm { T } _ { 2 }$ with a single interchain bond dominate, closely followed by I<sub>2</sub>, indicating that loops formed before the aggregation transition persist and are supplemented by interchain interactions. This reduces the relative fraction of all single-chain CT motifs, including I<sub>2</sub>. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-09/788fa88a-11a4-4082-b023-d6bf5fb6be7b/74859b504e3ae66434e1a86a104dc01249f9d69bc47599846d65ee84ddafb9ec.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-09/788fa88a-11a4-4082-b023-d6bf5fb6be7b/9f83bc417285b16d5eeffc95787a776f9c9e05881e5817b7f0ef050a0a654192.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-09/788fa88a-11a4-4082-b023-d6bf5fb6be7b/9ec412486ed9113f6312c1faaa2d28f003cd9d53846017e3ba9fb67e12784ef2.jpg)



T [K]


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-09/788fa88a-11a4-4082-b023-d6bf5fb6be7b/49ea2d8ae423feab8e01448666890f6228ca388882bbea3c30992405c8c80e44.jpg)



T [K]



FIG. 4. Average numbers of intrachain contacts $N _ { \mathrm { i n t r a } }$ (red circles) and interchain contacts $N _ { \mathrm { i n t e r } }$ (green squares), normalized by the total number of monomers m ⋅ n, together with their respective fractions $r _ { \mathrm { i n t r a } }$ (red circles) and $r _ { \mathrm { i n t e r } }$ (green squares), as functions of temperature T. (a) $N _ { \mathrm { i n t r a } }$ and $N _ { \mathrm { i n t e r } }$ for the 4-polymer system, with $N _ { \mathrm { i n t r a } }$ from the single-chain system (orange diamonds) shown for comparison; (b) $r _ { \mathrm { i n t r a } }$ and $r _ { \mathrm { i n t e r } }$ for the 4-polymer system; (c) $N _ { \mathrm { i n t r a } }$ and $N _ { \mathrm { i n t e r } }$ for the 8-polymer system; (d) $r _ { \mathrm { i n t r a } }$ and $r _ { \mathrm { i n t e r } }$ for the 8-polymer system. Black (inner) symbols denote values obtained from Eqs. (5) and (6), confirming the direct numerical calculations. For $\tau < T _ { c , }$ , no interchain contacts are present and $N _ { \mathrm { i n t r a } }$ grows steadily, while for $\begin{array} { r } { T \geq T _ { c } , } \end{array}$ all measures $( N _ { \mathrm { i n t r a } }$ $N _ { \mathrm { i n t e r } }$ , r<sub>intra</sub>, r<sub>inter</sub>) converge to constant values in both systems.


There are several notable differences between our results and previous studies on semiflexible polymers in which temperature is held constant while the Lennard-Jones interaction strength ϵ and polymer stiffness κ are varied.<sup>29,33</sup> In particular, those studies employed an implicit solvent model and did not include hydrophobic interactions. Within that framework, increasing ϵ effectively corresponds to decreasing temperature in our study. However, unlike our model, such an increase in interaction strength drives the coil-to-globule transition. Moreover, by increasing the stiffness and reducing the length of the individual polymers, such systems exhibit an isotropic-nematic transition wherein individual polymers align according to the system’s global director, forming bundles or tactoids. This directed rearrangement of polymers is accompanied by an enrichment in T<sub>3</sub>, L<sub>2</sub>, and I<sub>4</sub> motifs, which form prior to a possible chain collapse. In contrast, our system is dominated by the I<sub>3</sub>, I<sub>2</sub>, and T<sub>2</sub> motifs. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-09/788fa88a-11a4-4082-b023-d6bf5fb6be7b/6f3af3e7f5a009647b35bf439e6c271af9f835060306667b75d60b13d28957bf.jpg)



FIG. 5. Radius of gyration $R _ { 9 }$ (a) and degree of cohesion D<sub>c</sub> (b) as a function of temperature for the simulated models with 4 (red) and 8 (green) polymers. For reference, the single-chain values are also shown (black). $R _ { 9 }$ and D<sub>c</sub> are rescaled by the radius of gyration of the ideal chain $R _ { \mathfrak { g , i d e a l } } = b \sqrt { ( n - 1 ) / 6 }$ and system size L, respectively.


For the forced-coil system, however, the polymers aggregate at 350 K, a much higher temperature. Therefore, we conclude that the temperature of phase separation, driven by hydrophobic interactions, strongly depends on polymer conformation (290 vs 350 K). 

We note that for $T > T _ { \mathrm { c } } , D _ { \mathrm { c } } / L$ for the standard system is lower than that for the forced-globule system. This reflects the fact that forced-globular polymer chains cannot snake through the aggregate as flexible standard polymer chains do. A CT analysis reveals that this is indeed the case; the number of single-chain motifs and I<sub>2</sub> remains approximately constant for the entire temperature range, indicating that loops are stable and that for $T > T _ { \mathrm { { c } } }$ they are simply supplemented by interchain contacts, increasing only motifs of type $\mathrm { I } _ { 3 }$ and T<sub>2</sub>. 

Below $T _ { \mathrm { c } } ,$ where the solution is homogeneous, $R _ { \mathrm { g } }$ decreases by 15% as T increases from 240 to 280 K. Does this decrease in $R _ { \mathrm { g } }$ upon heating promote phase separation? In other words, if hypothetical polymer chains are used whose $R _ { \mathrm { g } }$ remains the same as that 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-09/788fa88a-11a4-4082-b023-d6bf5fb6be7b/e79d97d3886b3ae71c7993c7310594341ec14e784e92f438e1ad98278ac60993.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-09/788fa88a-11a4-4082-b023-d6bf5fb6be7b/6486816ce155e119c265a0250f9ece736e3f3ed46a761b999a8354ee17e5d866.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-09/788fa88a-11a4-4082-b023-d6bf5fb6be7b/cdcdc7c424d29605a89ac2588d2cc1d93d195c6af556e1bc0ff1c52bb98b7913.jpg)



for the standard system at 240 K, would the aggregation temperature be higher than $\mathit { T _ { c } ? }$ To answer this question directly, we would need to evaluate $T _ { \mathrm { c } }$ for the hypothetical polymer chains in water. Without such simulations, however, we propose that the answer is likely affirmative. First, as confirmed earlier and shown in Fig. 5(b), the aggregation temperature for the forced-coil system, which has a larger $R _ { \mathrm { g } }$ than the standard system, is 350 K, substantially higher than 290 K. This indicates that the aggregation temperature is highly sensitive to polymer conformation. Second, although $R _ { \mathrm { g } }$ decreases by only 15% from 240 to 280 K, the conformational change in the standard polymers over this temperature range is significant in terms of the number of intrachain contacts, $N _ { \mathrm { i n t r a } } ,$ and the corresponding CT motif, $I _ { 2 } ,$ as shown in Figs. 4 and $6 ( \mathrm { a } ) \mathrm { : } N _ { \mathrm { i n t r a } }$ increases by 152%, and $I _ { 2 }$ by 533%.



FIG. 6. Average number and average fraction of CT motifs as a function of temperature. Single-, double-, and triple-chain motifs are, respectively, colored red, purple, and green, while the quadruple-chain motif $\left( I _ { 4 } \right)$ is shown in orange. (a) and (c) CT motifs for, respectively, the 4 and 8 polymer systems. (b) and (d) CT fractions for, respectively, the 4 and 8 polymer systems. The structural phase transition at $T _ { c } = 2 9 0$ K (dashed line) is easily seen in all motif numbers.


In this study, we have investigated the structural and topological behavior of interacting hydrophobic polymer chains in aqueous solutions by combining molecular dynamics simulations with a CT analysis. By focusing on hard contacts in open chains, CT provided a robust framework for classifying and quantifying intraand interchain interactions, complementing conventional geometric descriptors such as radius of gyration and degree of aggregation. 

Our results reveal a temperature-driven collective structural transition, wherein homopolymer chains shift from isolated coil states to aggregated globule states. The temperature dependence of the radius of gyration $\left( R _ { \mathrm { g } } \right)$ in the multi-chain system shows that interchain hydrophobic interactions drive the complete collapse of polymer chains, coinciding with the onset of aggregation at the same temperature. Comparison of the standard system with two constrained systems, where interchain monomer–monomer interactions are identical, indicates that the aggregation temperature increases with greater chain extension. This suggests that the preliminary collapse of polymer chains (a 15% reduction in $R _ { \mathrm { g } }$ from 240 to 280 K) lowers the aggregation temperature compared to a hypothetical system where no reduction in $R _ { \mathrm { g } }$ occurs. 

This transition is captured not only by the standard geometric observables but also in the emergence and redistribution of CT motifs. We showed that CT motif enumeration not only recovers first-order descriptors (i.e., intra- and interchain contact fractions) through simple combinatorics but also affords a higher-order understanding of polymer organization inaccessible to traditional geometric metrics alone. In particular, the transition from predominantly intrachain motifs (S, P, X) at low temperatures to a significant population of multichain motifs $( \mathrm { T } _ { 2 } , \mathrm { L } _ { 2 } , \mathrm { T } _ { 3 } ,$ I , and ${ \mathrm { I } } _ { 4 } )$ at higher temperatures correlates directly with the emergence of aggregated states in the system. The sharp onset of interchain contacts and motifs should be interpreted cautiously. Observed in finite-size, explicit-solvent simulations, this behavior does not necessarily imply a breakdown of Ising-class criticality. It is possible that the polymer concentration is not sufficiently close to the critical concentration, or the temperature interval of 10 K is too large. To determine whether or not the step-like behavior persists in the thermodynamic limit, systematic finite-size scaling and/or numerical simulations of simpler model systems—such as lattice models, implicit-solvent models, or explicit-but-simplified solvent models—are required. 

The model polymer in this study is a freely jointed chain of hydrophobic monomers. Incorporating hydrophilic monomers or groups into the polymer chain enables the study of chemical inhomogeneity effects on collective phase transitions. The CT analysis-based approach, combined with explicit solvent-model MD simulations that are readily extensible to higher degrees of polymerization and non-linear chain architectures, facilitates the investigation of co-solvent effects, such as those from alcohols and electrolytes, on phase transitions in aqueous polymer solutions. This approach opens new avenues for exploring hierarchical phase transitions in soft matter, particularly in biologically relevant systems like protein condensation, RNA aggregation, and cellular phase-separated compartments. 

K.K. acknowledges JSPS KAKENHI (Grant Nos. 18KK0151, 20H02696, and 25K00969). Part of the computation was performed using the Research Center for Computational Science, Okazaki, Japan (Project Nos. 23-IMS-C112, 24-IMS-C106, and 25-IMS-C106). J.B. acknowledges Okayama University for its generous hospitality and the Research Foundation of DPhil Ragna Rask-Nielsen for funding the project. 

## AUTHOR DECLARATIONS

## Conflict of Interest

The authors have no conflicts to disclose. 

## Author Contributions

Junichi Komatsu: Data curation (equal); Formal analysis (equal); Investigation (equal); Methodology (equal); Software (equal); Validation (equal); Visualization (equal); Writing – original draft (equal). Kenichiro Koga: Conceptualization (equal); Funding acquisition (equal); Methodology (equal); Project administration (equal); Supervision (equal); Writing – review & editing (equal). Jonas Berx: Data curation (equal); Formal analysis (equal); Funding acquisition (equal); Investigation (equal); Methodology (equal); Project administration (equal); Software (equal); Visualization (equal); Writing – original draft (equal); Writing – review & editing (equal). 

## DATA AVAILABILITY

The data that support the findings of this study are available from the corresponding authors upon reasonable request. 

## REFERENCES



<sup>1</sup>E. Slagowski, B. Tsai, and D. McIntyre, “The dimensions of polystyrene near and below the theta temperature,” Macromolecules 9, 687–688 (1976). 





<sup>2</sup>M. Nierlich, J. P. Cotton, and B. Farnoux, “Observation of the collapse of a polymer chain in poor solvent by small angle neutron scattering,” J. Chem. Phys. 69, 1379–1383 (1978). 





<sup>3</sup>S.-T. Sun, I. Nishio, G. Swislow, and T. Tanaka, “The coil–globule transition: Radius of gyration of polystyrene in cyclohexane,” J. Chem. Phys. 73, 5971–5975 (1980). 





<sup>4</sup>K. Kubota, S. Fujishige, and I. Ando, “Single-chain transition of poly(Nisopropylacrylamide) in water,” J. Phys. Chem. 94, 5154–5158 (1990). 





<sup>5</sup>X. Wang, X. Qiu, and C. Wu, “Comparison of the coil-to-globule and the globuleto-coil transitions of a single poly(N-isopropylacrylamide) homopolymer chain in water,” Macromolecules 31, 2972–2976 (1998). 





<sup>6</sup>G. Zhang and C. Wu, “The water/methanol complexation induced reentrant coil-to-globule-to-coil transition of individual homopolymer chains in extremely dilute solution,” J. Am. Chem. Soc. 123, 1376–1380 (2001). 





<sup>7</sup>J. Hao, H. Cheng, P. Butler, L. Zhang, and C. C. Han, “Origin of cononsolvency, based on the structure of tetrahydrofuran-water mixture,” J. Chem. Phys. 132, 154902 (2010). 





<sup>8</sup>Y. Zhang, S. Furyk, D. E. Bergbreiter, and P. S. Cremer, “Specific ion effects on the water solubility of macromolecules: PNIPAM and the Hofmeister series,” J. Am. Chem. Soc. 127, 14505–14510 (2005). 





<sup>9</sup>R. L. Myers, A. Taira, C. Yan, S.-Y. Lee, L. K. Welsh, P. R. Ianiro, T. Yang, K. Koga, and P. S. Cremer, “Hydronium ions are less excluded from hydrophobic polymer–water interfaces than hydroxide ions,” J. Phys. Chem. B 129, 726–735 (2024). 





<sup>10</sup>Y. Maeda, T. Higuchi, and I. Ikeda, “Change in hydration state during the coil–globule transition of aqueous solutions of poly(N-isopropylacrylamide) as evidenced by FTIR spectroscopy,” Langmuir 16, 7503–7509 (2000). 





<sup>11</sup>K. Koga, “Osmotic second virial coefficient of methane in water,” J. Phys. Chem. B 117, 12619–12624 (2013). 





<sup>12</sup>B. Widom, P. Bhimalapuram, and K. Koga, “The hydrophobic effect,” Phys. Chem. Chem. Phys. 5, 3085 (2003). 





<sup>13</sup>C. Tanford, “The hydrophobic effect and the organization of living matter,” Science 200, 1012–1018 (1978). 





<sup>14</sup>H. Frauenkron and P. Grassberger, “Critical unmixing of polymer solutions,” J. Chem. Phys. 107, 9599–9608 (1997). 





<sup>15</sup>K. Binder, M. Müller, P. Virnau, and L. G. MacDowell, “Polymer + solvent systems: Phase diagrams, interface free energies, and nucleation,” in Advanced Computer Simulation, Advances in Polymer Science (Springer Berlin Heidelberg, 2005), pp. 1–110. 





<sup>16</sup>Y. Maki, T. Dobashi, and M. Nakata, “Kinetics of chain collapse in dilute polymer solutions: Molecular weight and solvent dependences,” J. Chem. Phys. 126, 134901 (2007). 





<sup>17</sup>M. Nakata, Y. Nakamura, N. Sasaki, and Y. Maki, “Transition from a chaincollapse process to a chain-aggregation process of poly(methyl methacrylate) in a mixed solvent,” Phys. Rev. E 76, 041805 (2007). 





<sup>18</sup>Y. Maki, T. Dobashi, and M. Nakata, “Comparison of the kinetics of chain aggregation and chain collapse in dilute polymer solutions,” Phys. Rev. E 78, 041802 (2008). 





<sup>19</sup>G. Graziano, “On the temperature-induced coil to globule transition of poly-N-isopropylacrylamide in dilute aqueous solutions,” Int. J. Biol. Macromol. 27, 89–97 (2000). 





<sup>20</sup>Y. Ding and G. Zhang, “Folding of a single polymer chain and phase transition,” Sci. Bull. 54, 1908–1911 (2009). 





<sup>21</sup>I. Hatano, K. Mochizuki, T. Sumi, and K. Koga, “Hydrophobic polymer chain in water that undergoes a coil-to-globule transition near room temperature,” J. Phys. Chem. B 120, 12127–12134 (2016). 





<sup>22</sup>C. Adams, The Knot Book (American Mathematical Society, Providence, RI, 2004). 





<sup>23</sup>N. C. H. Lim and S. E. Jackson, “Molecular knots in biology and chemistry,” J. Phys.: Condens. Matter 27, 354101 (2015). 





<sup>24</sup>A. Mashaghi, R. J. van Wijk, and S. J. Tans, “Circuit topology of proteins and nucleic acids,” Structure 22, 1227–1237 (2014). 





<sup>25</sup>A. Mugler, S. J. Tans, and A. Mashaghi, “Circuit topology of self-interacting chains: Implications for folding and unfolding dynamics,” Phys. Chem. Chem. Phys. 16, 22537–22544 (2014). 





<sup>26</sup>B. Scalvini, V. Sheikhhassani, J. Woodard, J. Aupic, R. T. Dame, R. Jerala, and ˇ A. Mashaghi, “Topology of folded molecular chains: From single biomolecules to engineered origami,” Trends Chem. 2, 609–622 (2020). 

