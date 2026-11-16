# Appele a chaque build (add_custom_target dans CMakeLists.txt) : ecrit git_version.hpp, seulement si son contenu change.
# SRC = repertoire des sources, OUT = fichier a ecrire.
# Version : "git describe --always --dirty" (hash court ; "-dirty" si des fichiers SUIVIS sont modifies,
# les fichiers non suivis ne sont pas comptes) ; "unknown" si git ou le depot est absent.
execute_process(COMMAND git -C "${SRC}" describe --always --dirty --abbrev=10
                OUTPUT_VARIABLE GIT_V OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET RESULT_VARIABLE GIT_R)
if(NOT GIT_R EQUAL 0 OR GIT_V STREQUAL "")
    set(GIT_V "unknown")
endif()
execute_process(COMMAND git -C "${SRC}" rev-parse --abbrev-ref HEAD
                OUTPUT_VARIABLE GIT_B OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET RESULT_VARIABLE GIT_BR)
if(NOT GIT_BR EQUAL 0 OR GIT_B STREQUAL "")
    set(GIT_B "unknown")
endif()
set(CONTENT "#define T2F_GIT_VERSION \"${GIT_V}\"\n#define T2F_GIT_BRANCH \"${GIT_B}\"\n")
set(OLD "")
if(EXISTS "${OUT}")
    file(READ "${OUT}" OLD)
endif()
if(NOT "${OLD}" STREQUAL "${CONTENT}")
    file(WRITE "${OUT}" "${CONTENT}")
endif()
