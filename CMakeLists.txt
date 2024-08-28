cmake_minimum_required(VERSION 3.8)

project(CV)

if(WIN32)
  # Enable Hot Reload for MSVC compilers if supported. conflict with address
  # sanitizer if ( POLICY CMP0141 ) cmake_policy( SET CMP0141 NEW )
  # set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
  # "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>"
  # ) endif()
  set(env "C:/Users/71958/AppData/Local/.xmake/packages") # packages location
  include_directories(
    "D:\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.38.33130\\include"
  ) # clangd cat't find it automatically

############ so you need to modify these two and xxx_ROOT below ############

  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /W3 -fsanitize=address /Zi")
elseif(LINUX)
  set(env "/home/vastina/.xmake/packages") # packages location
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
  set(CMAKE_CXX_FLAGS_DEBUG
      "${CMAKE_CXX_FLAGS_DEBUG} -Og -fsanitize=address -g")
endif()
cmake_policy(SET CMP0074 NEW) # for xxx_root
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(OpenCV_ROOT "${env}/o/opencv/4.10.0/d6c6c480e6414e34a4d6e6711fba5e82")
set(OpenCV_ROOT "${env}/o/opencv/4.10.0/localBuild")
# set(OpenCV_DIR "${OpenCV_ROOT}/x64/vc17/staticlib") # OpenCVConfig.cmake dir
find_package(OpenCV REQUIRED)
set(Qt6_ROOT "D:/Qt/6.7.2/msvc2019_64")
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets Network Multimedia Concurrent)

function(setup_project dir_name)
  # set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/src")
  set(SOURCE_DIR "${CMAKE_SOURCE_DIR}")
  set(${dir_name}_INCLUDE "${SOURCE_DIR}/${dir_name}/include")
  file(GLOB SOURCES_${dir_name} "${SOURCE_DIR}/${dir_name}/*.cpp")
  if(WIN32)
    set(_outname "${dir_name}")
  elseif(LINUX)
    set(_outname "${dir_name}.out")
  endif()
  add_executable(${_outname} ${SOURCES_${dir_name}})
  if(WIN32)
    set_property(TARGET ${_outname} PROPERTY CXX_STANDARD 20
                                             CXX_STANDARD_REQUIRED YES)
    set_target_properties(${_outname} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS} /utf-8")
    # set_target_properties(
    #   ${_outname}
    #   PROPERTIES WIN32_EXECUTABLE TRUE
    #              LINK_FLAGS "${LINK_FLAGS} /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
  endif()
  target_include_directories(
    ${_outname} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(
    ${_outname} PRIVATE
    ${OpenCV_LIBS}
    Qt6::Core Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Multimedia Qt6::Concurrent
  )
endfunction()

setup_project(open_image)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
