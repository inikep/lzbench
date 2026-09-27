# glza
CODECS += GLZA
GLZA_OBJS  := $(addprefix misc/glza/, GLZAcomp.o GLZAformat.o GLZAcompress.o GLZAencode.o GLZAdecode.o GLZAmodel.o)
GLZA_FLAGS := -std=gnu99
