function(set_project_arch_settings project_name)

        set(MSVC_ARCH_TYPE "Default" CACHE STRING "MSVC Architecture Selected")
        
        SET_PROPERTY(CACHE MSVC_ARCH_TYPE PROPERTY STRINGS "AVX" "AVX2" "AVX512") 

        
        if(MSVC)
                if(${MSVC_ARCH_TYPE} MATCHES "AVX")
                        set(PROJECT_ARCH_FEATURE_OPTION "/arch:AVX" /fp:fast)
                elseif(${MSVC_ARCH_TYPE} MATCHES "AVX2")
                        set(PROJECT_ARCH_FEATURE_OPTION "/arch:AVX2" /fp:fast)
                elseif(${MSVC_ARCH_TYPE} MATCHES "AVX512")
                        set(PROJECT_ARCH_FEATURE_OPTION "/arch:AVX512" /fp:fast)
                else()
                        set(PROJECT_ARCH_FEATURE_OPTION "")
                endif()
                
        else()
                set(PROJECT_ARCH_FEATURE_OPTION "-march=native" -ffast-math -m64)
        endif()

        target_compile_options(${project_name} INTERFACE ${PROJECT_ARCH_FEATURE_OPTION})
    
endfunction(set_project_arch_settings ${project_name})
