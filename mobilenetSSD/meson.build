project('mobilenetSSD_app', 'c', 'cpp',
        version : '3.22.0',
        default_options : [ 'warning_level=1',
                            'buildtype=debugoptimized',
                            'c_std=c11', 'cpp_std=c++17']
       )

# Compiler Arguments
compiler = meson.get_compiler('cpp')
compiler_version = compiler.version().split('.')[0].to_int()
if compiler_version >= 9
  message('GCC >= 9.0.0 detected, applying extra arguments.')
  add_global_arguments('-Wpessimizing-move', language : 'cpp')
  add_global_arguments('-Wredundant-move', language : 'cpp')
  add_global_arguments('-fconcepts', language : 'cpp')
endif

# tappas directories
relative_tappas_workspace =  '../dependencies'

# overlay paths
overlay_file = relative_tappas_workspace+'/core/hailo/plugins/overlay/overlay.cpp'

common_tensors_file = relative_tappas_workspace+'/core/hailo/libs/postprocesses/common/tensors.cpp'


# mobilenet postprocess paths
mobilenetSSD_file_1 = relative_tappas_workspace+'/core/hailo/libs/postprocesses/detection/mobilenet_ssd.cpp'




# include directories
rapidjson_inc = [include_directories(relative_tappas_workspace + '/core/open_source/rapidjson', is_system: true)]
hailo_general_inc = [include_directories(relative_tappas_workspace + '/core/hailo/general')]
postproccess_inc = [include_directories(relative_tappas_workspace + '/core/hailo/libs/postprocesses'),
 include_directories(relative_tappas_workspace + '/core/hailo/libs/postprocesses/detection'),
 include_directories(relative_tappas_workspace + '/core/hailo/plugins/common'), 
 include_directories(relative_tappas_workspace + '/core/hailo/plugins'), 
 include_directories(relative_tappas_workspace + '/core/hailo/plugins/overlay')
  ]


# Xtensor Include Directories
xtensor_base_inc = include_directories(get_option('libxtensor'), is_system: true)
xtensor_inc = [xtensor_base_inc]
if get_option('include_blas')
  xtensor_blas_inc = include_directories(get_option('libblas'), is_system: true)
  xtensor_inc += [xtensor_blas_inc]
endif

# find opencv and hailort
opencv_dep = dependency('opencv4', version : '>= 4.0', method : 'pkg-config')
hailort_lib = compiler.find_library('hailort')


executable('mobilenetSSD_app',
    
    [overlay_file, 'mobilenetSSD_app.cpp',mobilenetSSD_file_1,common_tensors_file],
    gnu_symbol_visibility : 'default',
    dependencies : [opencv_dep, hailort_lib],
    include_directories: [rapidjson_inc, hailo_general_inc, postproccess_inc]+ xtensor_inc,
    link_args : '-lpthread',
)

