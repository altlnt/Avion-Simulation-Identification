# identification_modele_avion
identification_modele_avion collab A. LETALENET M. ALLICHE

Différents scirpts : 
Coef_aero.ipynb = Décrit les différentes étapes pour générer les équations des coefficients aérodynamiques Cd et Cl, ainsi que leur lambdification. 

Force_aerodynamique.ipynb = Décrit les différentes étapes de génération des équations pour les vecteurs des forces aérodynamiques dans le repère inertielle. Permet aussi de lambdifier ces équations dans le cas de 5 surfaces portantes et avec 4 moteurs tournant à la même vitesse. 

equation_dynamique.ipynb = Génère les équations de la dynamique lambdifier, c'est à dire qu'il donne l'accélération angulaire et l'accélération en fonction des forces et couples. Ces fonctions ne sont pas utilisés dans ce simulateur. 

equation_generator.ipynb = Regroupe les différents scripts décrit ci-dessus et adapte les équations pour une géométrie particulière. Génère ensuite les équations lambdifier, et les exportent vers le fichier "fichier_fonction". Ce fichier est la liste suivante : 
    # 0 : VelinLDPlane_function
    # 1 : dragDirection_function
    # 2 : liftDirection_function
    # 3 : compute_alpha_sigma()
    # 4 : Coeff_function qui calcul les coeffs aéro pour toutes les surfaces tel que [Cl, Cd]
    # 5 : Effort_Aero qui renvoi un liste tel que [Force, Couple]
        

MoteurPhysique_class.py = Exploite les fonctions du fichier de fonctions produit par le script "equation_generator.ipynb", pour reconstruire la dynamique du drone. Il permet aussi de sauvegarder les différentes grandeurs souhaiter dans un fichier de log. 

Gui_class.py = Cette classe génère les fenêtres graphiques qui vont permettre d'observer le drone lors de la simulation en temps réel. De plus il gère aussi les entrées envoyées depuis le joystick. 

Simulator_Class.py = Cette classe fait le lien entre le moteur physique et la classe GUI qui gère la partie graphique et les entrées. Dans un premier temps, on récupère les entrées du joystick avec la classe GUI, qui sont ensuité injectées dans le moteurs physiques qui va alors données les nouvelles positions et orientations qui servirons à alimenter l'interfaces graphiques. 



Pour lancer une simulation, dans un premier temps il est nécessaire de lancer le scripts "equation_générator.ipynb" et vérifier que le fichier de fonctions a bien été écrit. Ensuite, il suffit de brancher une manette et de lancer le fichier "Simulator_Class.py", la simulation va se lancer, et pendant un certain temps (par défault 5s) le drone va avancer sans ressentir de forces (ce temps est inscrit dans le moteur physique dans le calcul des forces). 
Par la suite, les forces vont agir et on pourra le piloter manuellement. Il est possible de changer les paramètres physiques du drones dans la classe du moteur physique (seules les valeurs des aires sont à modififier dans le script "equation_generator" et le relancer pour avoir les équations correspondantes). Une fois fait, on peut relancer la simulation directement. 

Pour le pilotatge manuel : 
	- joystick gauche horizontale = roll
	- joystick gauche vertical = pitch
	- joystick droite horizontale = yaw
 	- joystick drotie vericale = vitesse des moteurs
Si le drone n'est pas assez réactif, ou à l'inverse il l'est trop, il suffit d'ajuster les gains des ailerons de controle (k0, k1, k2) ou les inerties correspondantes.



