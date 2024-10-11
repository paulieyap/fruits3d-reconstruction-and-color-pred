.PHONY : all latex bibtex pdf view clean distclean

################################################################################
##################### CHANGE for each student/supervisor #######################
################################################################################
# Name of the master student
STUDENT="Jeanne Paulie Yap"
# Title of the thesis
THESIS="[CHANGE] Blah-based blaherry for real-time blah"
# Name of cosupervisor who is not cyrill, i.e. Andres, Jens, Olga, Nacho, 
# Rhiney, Lorenzo, Alfred, Nived, Philipp... Ok, maybe not Alfred.
COSUPERVISOR="[CHANGE] Alfred Alfredson" 
################################################################################


################################################################################
################################ DO NOT CHANGE #################################
################################################################################

TARGET=task
SOURCE=$(TARGET).tex

all: latex1 bib latex2 latex3
	
latex1 latex2 latex3:
	pdflatex "\def\student{$(STUDENT)} \
						\def\thesis{$(THESIS)} \
						\def\cosupervisor{$(COSUPERVISOR)} \
						\input{$(SOURCE)}"

bib:
	bibtex $(TARGET)

view:
	open $(TARGET).pdf &

gview:
	gnome-open $(TARGET).pdf &

clean:
	rm -f $(TARGET).aux $(TARGET).bbl $(TARGET).blg $(TARGET).log $(TARGET).fls $(TARGET).fdb_latexmk $(TARGET).synctex.gz

distclean:clean
	rm -f $(TARGET).pdf 

################################################################################