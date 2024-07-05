set (MODULE_NAME voro)

# Get all compilation units.
#file (GLOB ${MODULE_NAME}_SRCS *.cc)
file (GLOB ${MODULE_NAME}_SRCS voro++.cc)

# Create the library.
add_library (${MODULE_NAME} ${${MODULE_NAME}_SRCS})

install (TARGETS ${MODULE_NAME} 
		ARCHIVE DESTINATION lib 
		LIBRARY DESTINATION lib)
