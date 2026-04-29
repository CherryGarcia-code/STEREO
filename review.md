Reviewer #1: The manuscript titled "Opponent ventrolateral striatal circuits regulate behavioral flexibility and rigidity" by Gonzales, Shalom, and colleagues addresses an important gap in our understanding of the circuit level mechanisms underlying behavioral adaptation and maladaptation. The authors employ a sophisticated combination of behavioral deep learning analysis, calcium imaging (GCaMP), and neural modulation approaches (DREADDs and optogenetics) to dissect ventrolateral striatal (VLS) contributions to flexible versus rigid behavioral states.
The study leverages high dose cocaine administration as an inducible model of behavioral rigidity and reports a robust activation of VLS neural ensembles, as indicated by cFos expression. Bidirectional modulation of licking behavior is demonstrated through chemogenetic inhibition of spiny projection neurons (SPNs), while optogenetic activation of direct pathway SPNs successfully recapitulates the cocaine induced licking phenotype. These findings compellingly suggest that heightened direct pathway VLS activity is sufficient to drive the observed maladaptive behavioral output. The authors further incorporate a sucrose splash assay as a non drug stimulus to test flexible reward seeking behavior, adding breadth and translational value to their behavioral paradigm. In this case, activation of the indirect pathway suppresses the adaptive licking behavior. Taken together, I agree with the authors suggestion that the results align with a competitive model of striatal function, wherein the balance of SPN output across pathways governs behavioral selection and suppression—a framework that the data support convincingly.
The manuscript is clearly written, methodologically rigorous, and presents timely, novel insights into striatal circuitry.
Major Comments
* The manuscript does not specify whether cFos immunohistochemistry was performed following a single cocaine exposure or after repeated exposures. This distinction is important because repeated high dose cocaine could potentially narrow cFos responsive neuronal populations, paralleling the progressive behavioral rigidity. Clarification here would also help contextualize how these findings relate to the GCaMP activity patterns reported across sessions.
Minor Comments
* The Introduction emphasizes the concept of "involuntary, rigid actions." It would be helpful to address whether the orofacial licking behavior examined here truly fits the authors' definition of involuntary behavior, or whether it should instead be framed as a stereotyped yet volitional action.
* Provide the full strain name and JAX stock number for all mouse lines used, to ensure reproducibility.
* The persistence of cocaine related behavioral phenotypes 24 hours after the final exposure suggests a lasting reorganization of behavioral output. Expanding on this point could reinforce the significance of VLS plasticity in maladaptive decision making.
* In Figure 2I, the density estimation plot for 30 mg/kg cocaine appears visually similar to the plot in Figure 2H, which contains both 20 mg/kg and 30 mg/kg data. Clarifying the distinction between these figures or adjusting the visual presentation may prevent confusion.



Reviewer #2: This study asks how basal ganglia circuits drive a transition from flexible action selection to compulsive, repetitive behavior. To quantify this shift, the authors introduce STEREO, a machine-learning framework that extracts high-resolution, semantically meaningful descriptions of natural mouse movement from video. They show that repeated cocaine exposure compresses a normally diverse behavioral repertoire into rigid, stereotyped routines—most notably persistent surface licking. Combining optogenetics with fiber photometry, they argue that ventrolateral striatum (VLS) functions as an opponent control system: behavioral flexibility reflects the moment-to-moment balance between activity in direct- versus indirect-pathway populations within VLS. This is great approach. I have a few concerns and suggestions that may help strengthen the manuscript.

1. A key limitation is that the dorsomedial striatum (DMS) is largely absent from the study's real-time recordings and causal tests. Although the authors' c-Fos mapping indicates that DMS activity remains elevated at the 30 mg/kg cocaine dose, their photometry and optogenetic manipulations focus almost exclusively on VLS. Without recording from, or perturbing, DMS during the period of "behavioral collapse," the claim that VLS "dominates" the competitive landscape is not directly demonstrated. An alternative interpretation that DMS contributes critically to sustaining the rigid state but is simply unmeasured remains viable.

2. The behavioral readout also biases the anatomical conclusion. Because surface licking is an orofacial action with strong known links to VLS, the central result can feel partially predetermined: choosing a licking-dominant stereotypy naturally highlights VLS involvement. If the authors had emphasized limb rigidity, axial stereotypies, or other repetitive motor patterns, a different striatal subregion might have emerged as most predictive and causally potent. This raises the possibility that VLS is not a general-purpose "compulsivity hub," but rather a specialized node controlling a constrained set of motor routines. Other types of behavioral rigidity can be measured and correlated with the VLS or DMS activity.

3. While the authors use extensive optogenetic and chemogenetic manipulations to argue that VLS activity is necessary and sufficient for rigidity, they do not perform parallel loss-of-function tests in DMS during the rigid state. That omission leaves a logical gap: if DMS remains active under high-dose cocaine, it may still be required to maintain rigidity even if VLS can bias the expressed output. Without directly disrupting DMS during rigidity, the study cannot conclusively show that control has shifted from DMS to VLS, only that VLS provides a powerful handle for modulating the observed orofacial stereotypy.



Reviewer #3: In this straightforward & interesting paper, Gonzalez et al developed a deep learning pipeline to classify spontaneous mouse behavior (e.g. wall licking, body licking, locomotion, etc. They find that repeated exposure to cocaine activates Fos in the ventrolateral striatum and induces licking behavior. Indirect-pathway activation in that subregion alleviates the phenotype and produces other actions, such as locomotion, while direct-pathway activation is sufficient to generate licking behavior.

I enjoyed the paper due to the dramatic behavioral phenotypes generated by their optogenetic manipulations and by the drug pairings.

--

Please include screenshots of the videos (and also potentially links to videos) to give the reader a sense of the type and quality of data included in the classifier, including examples of what the different classified behaviors actually look like by eye. The match between the 2 observers isn't high, suggesting the videos themselves might not be too high quality.

In 2I, it doesn't look super clear that VLS activation is greater than DMS activation. It looks like a band of activation that spans both regions. Is that one example animal? Can the average across mice be shown as well as the example animal, also for the 20mg/kg condition?

Is the licking behavior induced by repeated cocaine exposure specific for the chamber that had been paired with cocaine?

Re code release- the provided github link doesn't seem to work. Moreover, having a separate code release for the classifier vs data analysis for the figure panels would be preferable / more useful.


Minor

Is surface licking the same as wall licking? Please use consistent terminology throughout the paper.



Reviewer #4: In their manuscript entitled "Opponent ventrolateral striatal circuits regulate behavioral flexibility and rigidity," Gonzales et al. combine an impressive and technically sophisticated set of approaches—including deep-learning-based behavioral classification (STEREO), population calcium imaging, and pathway-specific circuit manipulations in freely behaving mice—to investigate how basal ganglia circuits regulate the balance between adaptive flexibility and behavioral rigidity. By integrating high-resolution quantification of naturalistic behavior with causal circuit interrogation, the authors provide a comprehensive analysis of ventrolateral striatal pathways controlling orofacial action selection. Their results reveal an opponent organization of direct and indirect pathway activity that links physiological circuit dynamics to both flexible behavioral repertoire and psychostimulant-induced behavioral collapse. The study is ambitious, methodologically rigorous, and conceptually well framed, bringing together computational ethology and systems neuroscience to address a fundamental problem in basal ganglia function. Overall, the dataset is rich, the conclusions are supported by convergent evidence, and the work provides important insight into the circuit mechanisms that underlie both adaptive behavioral flexibility and pathological rigidity. I have a series of comments and suggestions that I hope will help further strengthen the manuscript.

Abstract:
In the last sentence, the expression "biased recruitment of basal ganglia circuit" could be clarified to indicate in which direction this bias occurs.

Results:
Automated resolution of mouse behavior using STEREO
The authors should clarify in the Results or Methods section why they chose not to use the top‑view camera to train their algorithm. Regarding the validation of the STEREO method, the authors provide an overall accuracy and an F1 score, but it is unclear whether these values correspond to human-human comparison or human-algorithm comparison. In addition, the authors should explicitly compare human-human and human-algorithm agreement, for instance in a supplementary figure. This comparison could also be performed for each label/behavior.
The readability of Figures 1E-F could be improved by adding behavior labels on at least one axis.

Repeated exposure to cocaine engages the ventrolateral striatum and progressively leads to maladaptive action selection
To give more weight to the observations "mice exhibited a rich and flexible behavioral repertoire" and "behavior collapsed into a dominant and persistent pattern", the authors could expand their behavioral quantifications, for example by quantifying the transition rate between behaviors or the number of distinct actions per unit of time.
The way the data are displayed in Figures S2H-M makes it difficult to clearly appreciate the authors' observations, particularly regarding the transient effects on iSPNs.
Figure 2L appears to show a stronger activation of dSPNs after lick onset. Is this effect significant, or is it largely explained by differences in lick‑bout duration between dSPN‑ and iSPN‑recorded animals?
While it makes sense that the authors focused on surface licking behaviors, the results could benefit from being expanded to other behaviors affected (or not) by repeated cocaine exposure, such as locomotion, grooming, or rearing.
The curves plotted in Figures 2E-F are difficult to distinguish. Perhaps stacked graphs (as in Figure 3E) would improve readability.
The c‑Fos labeling in Figure 2H is difficult to fully appreciate. Adding higher‑magnification images at the level of the VLS may help.
For the fiber‑photometry experiments (and also the optogenetic and DREADD experiments), histological validation of viral transfection and fiber placement is missing.

VLS indirect pathway activation alleviates cocaine-induced behavioral rigidity
During the optogenetic stimulation iSPN, is there any effect on the other behaviors other than locomotion and floor licking? To complement the results provided in Figures 3J-K, is there any change in the transition probabilities between action, similar to Figure 2G

Opponent regulation of cocaine-induced behavioral rigidity by VLS direct and indirect pathway inhibition
I am not sure I fully understand the conclusions drawn from the data presented in Figures 4H-I. While the plots clearly show that mice velocity decreases following CNO administration during floor licking, it remains unclear how the authors distinguish between locomotive licking and non-locomotive licking in their analysis.
I am also wondering about the classification into locomotion/stationary. Figure 4I (and figure 3M) shows a relatively large number of bins for velocity values below 0.5 cm/s. Shouldn't these epochs belong to the stationary class?
DREADD activation of iSPN produced an effect opposite to the inhibition for the time spent in surface licking. Did the authors also observe opposite effects for other parameters that showed significant effect after the inhibition of iSPN?

VLS direct pathway activation drives cocaine-mimetic behavioral rigidity in drug-naïve mice
The analyses displayed should be complemented with additional behaviors, particularly to support the authors' claim that some "selected behaviors persisted throughout the stimulation epoch".
The temporal analyses displayed in Figures 5G and 5I suggest evolving dynamics, with grooming probability increasing then plateauing, while floor licking probability decreases over time. Do the authors have additional insight into these temporal dynamics?
The conclusion that dSPN activation alone induces rigidity in action‑selection patterns is mostly supported by increased time spent grooming and licking. I feel this conclusion could be strengthened by analyzing additional variables, such as bout duration or transition probabilities between behaviors.

Adaptive action selection is associated with VLS direct and indirect athway activity and impaired by indirect pathway activation
For the fiberphotometry experiments, why did the authors divide the episodes in quartiles of bout duration instead of reusing the same time-warp method from Figure 2M? And what about the neuronal activity during grooming episodes?
In these experiments, it seems that for the longest experiments (Q4 of bout duration) the activity in dSPN decreases during the ongoing bout, while iSPN activity remains consistently high Is there an actual significant difference?
The authors report that stimulation of iSPN suppresses body licking, but it seems that grooming is also suppressed. To what extent can this potential effect be in line with the increase in grooming induced by dSPN activation?

Methods:
Behavioral assays
First line, typo A2a-cre instead of Drd2 cre?
The dimensions of the optic fibers used is missing, as well as the light source used.
Stereotactic surgeries
The volumes delivered and the exact type of the AAVs used are missing. The postoperative recovery duration is also missing.
Fiber photometry acquisition
For preprocessing, the authors applied a first step of de-bleaching before fitting the signal channel to the isosbestic channel that I'm not used to. What is the rationale behind this approach and could the authors provide references to studies that used this method?
The first couple of sentences in the second paragraph in the 'fiber photometry analysis' subsection are somewhat unclear. Please rephrase
Behavioral analysis -STEREO
In equation 2, typo: 'frame' instead of 'fram'
It unclear whether the training of the algorithm was performed across all conditions (especially saline vs cocaine) or only with saline recordings
I would advise to replace the label 'jump' by something like 'cut' or 'no mouse' as it may be misleading and could be understood as the mouse jumping
In this section, and possibly throughout the manuscript, the three labels 'undefined', 'back to camera' and 'unknown' seem to refer to the same class. The authors should consider using a single label.
I'm surprised by the clear separation in the velocity histogram between immobility and locomotion (Figure S1F). Perhaps I am misreading the units, but does it make sense to classify velocities in the range of 10^-2 cm/sec as locomotion?