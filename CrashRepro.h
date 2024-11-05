#pragma once

#include <winsdkver.h>
#define _WIN32_WINNT 0x0A00
#include <sdkddkver.h>

#define NODRAWTEXT
//#define NOGDI
#define NOBITMAP
#define NOMCX
#define NOSERVICE
#define NOHELP
#ifndef NOMINMAX
#define NOMINMAX
#endif

// ==================== STL ====================
#include <assert.h>
#include <iostream>
#include <fstream>
#include <shlobj.h>
#include <type_traits>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <map>

// ==================== DX 12 ====================
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "dxguid.lib")
#pragma comment(lib, "RuntimeObject.lib")

#define D3D12_GPU_VIRTUAL_ADDRESS_NULL      ((D3D12_GPU_VIRTUAL_ADDRESS)0)
#define D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN   ((D3D12_GPU_VIRTUAL_ADDRESS)-1)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <wrl.h>
#include <wrl/client.h>
#include <wrl/event.h>

#include <d3d12.h>
#include <d3dx12.h>
#include <dxgi1_6.h>
#include <dxgidebug.h>

#include <pix3.h>

using string = std::string;
using wstring = std::wstring;

typedef unsigned char      u8;
typedef unsigned short int u16;
typedef unsigned int       u32;
typedef unsigned long long u64;
typedef signed char        i8;
typedef signed short int   i16;
typedef signed int         i32;
typedef signed long long   i64;
typedef wchar_t            wchar;
typedef unsigned __int64   uintptr;
