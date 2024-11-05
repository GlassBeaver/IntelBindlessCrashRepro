#include "CrashRepro.h"

#include <windows.h>
#include <wrl.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h> // Not used now, but keep if other shaders need in-program compilation
#include <dxcapi.h>      // Not needed since shader is loaded from file
#include <cstdint>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>

using Microsoft::WRL::ComPtr;

const UINT BufferCount = 256;
const UINT BufferSize = 1024;

// Helper function to load a compiled shader binary from file
std::vector<char> LoadShaderBinary(const std::wstring& filename)
{
	wstring filepath = filename;
	if (!std::filesystem::exists(filepath))
		filepath = L"x64\\Debug\\" + filename;
	assert(std::filesystem::exists(filepath));

	std::ifstream file(filepath, std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
		__debugbreak();
		exit(-1);
	}

	std::streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);

	std::vector<char> buffer(size);
	if (!file.read(buffer.data(), size)) {
		__debugbreak();
		exit(-1);
	}

	return buffer;
}

void CreateBuffers(ID3D12Device* device,
	std::vector<ComPtr<ID3D12Resource>>& buffers,
	std::vector<ComPtr<ID3D12Resource>>& uploadBuffers,
	ID3D12DescriptorHeap* SVRDescriptorHeap,
	ID3D12GraphicsCommandList* CommandList)
{
	D3D12_HEAP_PROPERTIES uploadHeapProperties = {};
	uploadHeapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;

	D3D12_HEAP_PROPERTIES normalHeapProperties = {};
	normalHeapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;

	D3D12_RESOURCE_DESC uploadBufferDesc = {};
	uploadBufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
	uploadBufferDesc.Width = BufferSize;
	uploadBufferDesc.Height = 1;
	uploadBufferDesc.DepthOrArraySize = 1;
	uploadBufferDesc.MipLevels = 1;
	uploadBufferDesc.Format = DXGI_FORMAT_UNKNOWN;
	uploadBufferDesc.SampleDesc.Count = 1;
	uploadBufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

	D3D12_RESOURCE_DESC normalBufferDesc = {};
	normalBufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
	normalBufferDesc.Width = BufferSize;
	normalBufferDesc.Height = 1;
	normalBufferDesc.DepthOrArraySize = 1;
	normalBufferDesc.MipLevels = 1;
	normalBufferDesc.Format = DXGI_FORMAT_UNKNOWN;
	normalBufferDesc.SampleDesc.Count = 1;
	normalBufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
	normalBufferDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

	uploadBuffers.resize(BufferCount);
	buffers.resize(BufferCount);

	const u32 svrDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

	for (UINT i = 0; i < BufferCount; ++i) {
		assert(SUCCEEDED(device->CreateCommittedResource(
			&uploadHeapProperties,
			D3D12_HEAP_FLAG_NONE,
			&uploadBufferDesc,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&uploadBuffers[i])
		)));

		assert(SUCCEEDED(device->CreateCommittedResource(
			&normalHeapProperties,
			D3D12_HEAP_FLAG_NONE,
			&normalBufferDesc,
			D3D12_RESOURCE_STATE_COMMON,
			nullptr,
			IID_PPV_ARGS(&buffers[i])
		)));

		D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
		uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
		uavDesc.Format = DXGI_FORMAT_UNKNOWN;
		uavDesc.Buffer.FirstElement = 0;
		uavDesc.Buffer.NumElements = BufferSize / sizeof(u32);
		uavDesc.Buffer.StructureByteStride = sizeof(u32);

		CD3DX12_CPU_DESCRIPTOR_HANDLE uavDescHandle(SVRDescriptorHeap->GetCPUDescriptorHandleForHeapStart(), i, svrDescriptorSize);
		device->CreateUnorderedAccessView(buffers[i].Get(), nullptr, &uavDesc, uavDescHandle);

		u32* bufferData = nullptr;
		uploadBuffers[i]->Map(0, nullptr, reinterpret_cast<void**>(&bufferData));

		for (UINT j = 0; j < BufferSize / sizeof(u32); ++j)
			bufferData[j] = j;

		uploadBuffers[i]->Unmap(0, nullptr);

		CommandList->CopyResource(buffers[i].Get(), uploadBuffers[i].Get());
	}
}

void ExecuteComputeShader(
	ID3D12Device* device,
	ID3D12CommandQueue* commandQueue,
	ID3D12GraphicsCommandList* commandList,
	std::vector<ComPtr<ID3D12Resource>>& buffers,
	ID3D12PipelineState* computePipelineState,
	ID3D12RootSignature* rootSignature,
	ID3D12DescriptorHeap* SVRDescriptorHeap)
{
	commandList->SetPipelineState(computePipelineState);
	commandList->SetComputeRootSignature(rootSignature);
	commandList->SetDescriptorHeaps(1, &SVRDescriptorHeap);

	for (UINT i = 0; i < BufferCount; ++i)
	{
		D3D12_GPU_VIRTUAL_ADDRESS bufferAddress = buffers[i]->GetGPUVirtualAddress();
		commandList->SetComputeRoot32BitConstant(0, i, 0);
		commandList->Dispatch(1, 1, 1);
	}

	D3D12_RESOURCE_BARRIER barrier{};
	barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
	barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
	barrier.UAV.pResource = nullptr;
	commandList->ResourceBarrier(1, &barrier);

	assert(SUCCEEDED(commandList->Close()));
	ID3D12CommandList* commandLists[] = { commandList };
	commandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);

	// Wait for GPU to finish execution
	ComPtr<ID3D12Fence> fence;
	device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
	HANDLE fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
	commandQueue->Signal(fence.Get(), 1);
	if (fence->GetCompletedValue() < 1) {
		fence->SetEventOnCompletion(1, fenceEvent);
		WaitForSingleObject(fenceEvent, INFINITE);
	}
	assert(CloseHandle(fenceEvent) != 0);
}

void DownloadAndCheckBuffers(ID3D12Device* device, ID3D12CommandQueue* commandQueue, std::vector<ComPtr<ID3D12Resource>>& buffers) 
{
	D3D12_HEAP_PROPERTIES heapProperties = {};
	heapProperties.Type = D3D12_HEAP_TYPE_READBACK;

	D3D12_RESOURCE_DESC bufferDesc = {};
	bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
	bufferDesc.Width = BufferSize;
	bufferDesc.Height = 1;
	bufferDesc.DepthOrArraySize = 1;
	bufferDesc.MipLevels = 1;
	bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
	bufferDesc.SampleDesc.Count = 1;
	bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

	ComPtr<ID3D12Fence> fence;
	assert(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))));
	HANDLE fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

	for (UINT i = 0; i < BufferCount; ++i)
	{
		ComPtr<ID3D12Resource> readbackBuffer;
		assert(SUCCEEDED(device->CreateCommittedResource(
			&heapProperties,
			D3D12_HEAP_FLAG_NONE,
			&bufferDesc,
			D3D12_RESOURCE_STATE_COPY_DEST,
			nullptr,
			IID_PPV_ARGS(&readbackBuffer)
		)));

		ComPtr<ID3D12CommandAllocator> commandAllocator;
		assert(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocator))));

		ComPtr<ID3D12GraphicsCommandList> commandList;
		assert(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocator.Get(), nullptr, IID_PPV_ARGS(&commandList))));

		commandList->CopyResource(readbackBuffer.Get(), buffers[i].Get());
		assert(SUCCEEDED(commandList->Close()));
		ID3D12CommandList* commandLists[] = { commandList.Get() };
		commandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);

		assert(SUCCEEDED(commandQueue->Signal(fence.Get(), i + 1)));
		if (fence->GetCompletedValue() < i + 1)
		{
			fence->SetEventOnCompletion(1, fenceEvent);
			WaitForSingleObject(fenceEvent, INFINITE);
		}

		u32* mappedData = nullptr;
		readbackBuffer->Map(0, nullptr, reinterpret_cast<void**>(&mappedData));
		assert(mappedData);

		for (UINT j = 0; j < BufferSize / sizeof(u32); ++j)
			if (mappedData[j] != j + 1)
				printf("Buffer %d, Element %d is incorrect. Expected: %d but got: %d\n", i, j, j + 1, mappedData[j]);

		readbackBuffer->Unmap(0, nullptr);
	}

	assert(CloseHandle(fenceEvent) != 0);
}

void EnableDebugLayer()
{
	ComPtr<ID3D12Debug> debugInterface;
	assert(SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugInterface))));
	debugInterface->EnableDebugLayer();
	ComPtr<ID3D12Debug1> debugInterface1;
	if (SUCCEEDED(debugInterface.As(&debugInterface1)))
		debugInterface1->SetEnableGPUBasedValidation(TRUE);
}

static wstring GetLatestWinPixGpuCapturerPath_Cpp17()
{
	wchar* programFilesPath = nullptr;
	SHGetKnownFolderPath(FOLDERID_ProgramFiles, KF_FLAG_DEFAULT, NULL, &programFilesPath);

	std::filesystem::path pixInstallationPath = programFilesPath;
	pixInstallationPath /= "Microsoft PIX";

	wstring newestVersionFound;

	for (auto const& directory_entry : std::filesystem::directory_iterator(pixInstallationPath))
		if (directory_entry.is_directory())
			if (newestVersionFound.empty() || newestVersionFound < directory_entry.path().filename().c_str())
				newestVersionFound = directory_entry.path().filename().c_str();

	assert(!newestVersionFound.empty());

	return pixInstallationPath / newestVersionFound / L"WinPixGpuCapturer.dll";
}

ComPtr<IDXGIAdapter4> GetAdapter(D3D_FEATURE_LEVEL& OutMaxFeatureLevel, const char* GPUIHV, bool bUseWarp)
{
	ComPtr<IDXGIFactory6> DXGIFactory;

	u32 createFactoryFlags = 0;
#if defined(DEBUG) || defined(DECKDEBUG)
	createFactoryFlags = DXGI_CREATE_FACTORY_DEBUG;
#endif

	assert(SUCCEEDED(CreateDXGIFactory2(createFactoryFlags, IID_PPV_ARGS(&DXGIFactory))));

	ComPtr<IDXGIAdapter1> dxgiAdapter1;
	ComPtr<IDXGIAdapter4> dxgiAdapter4;
	u64 maxDedicatedVideoMemory = 0;

	if (bUseWarp)
	{
		assert(SUCCEEDED(DXGIFactory->EnumWarpAdapter(IID_PPV_ARGS(&dxgiAdapter1))));
		assert(SUCCEEDED(dxgiAdapter1.As(&dxgiAdapter4)));

		DXGI_ADAPTER_DESC1 desc;
		assert(SUCCEEDED(dxgiAdapter1->GetDesc1(&desc)));
		maxDedicatedVideoMemory = desc.DedicatedVideoMemory;

		if (SUCCEEDED(D3D12CreateDevice(dxgiAdapter1.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr)))
			OutMaxFeatureLevel = D3D_FEATURE_LEVEL_12_0;
		else
			assert(false);

		assert(dxgiAdapter1);
		assert(SUCCEEDED(dxgiAdapter1.As(&dxgiAdapter4)));
	}
	else
	{
		u32 iAdapter = 0;
		DXGI_ADAPTER_DESC1 adapterDesc;

		enum class EForceGPU { None, NVIDIA, AMD, Intel };
		EForceGPU forceGPU = EForceGPU::None;

		if (GPUIHV)
			if (_stricmp(GPUIHV, "NVIDIA") == 0)
				forceGPU = EForceGPU::NVIDIA;
			else if (_stricmp(GPUIHV, "AMD") == 0)
				forceGPU = EForceGPU::AMD;
			else if (_stricmp(GPUIHV, "Intel") == 0)
				forceGPU = EForceGPU::Intel;
			else
				assert(false);

		for (u32 i = 0; DXGIFactory->EnumAdapters1(i, &dxgiAdapter1) != DXGI_ERROR_NOT_FOUND; i++)
		{
			DXGI_ADAPTER_DESC1 desc;
			dxgiAdapter1->GetDesc1(&desc);
			printf("GPU #%d: %ws\n", i, desc.Description);
		}

		if (forceGPU == EForceGPU::None)
		{
			if (SUCCEEDED(DXGIFactory->EnumAdapterByGpuPreference(iAdapter, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&dxgiAdapter1))))
			{
				// no IHV is forced: pick the preferred high-performance adapter
				DXGI_ADAPTER_DESC1 desc;
				dxgiAdapter1->GetDesc1(&desc);
				maxDedicatedVideoMemory = desc.DedicatedVideoMemory;
				adapterDesc = desc;
			}
			else
			{
				// no IHV is forced: find the adapter with the most amount of VRAM
				for (u32 i = 0; DXGIFactory->EnumAdapters1(i, &dxgiAdapter1) != DXGI_ERROR_NOT_FOUND; i++)
				{
					DXGI_ADAPTER_DESC1 desc;
					dxgiAdapter1->GetDesc1(&desc);
					const u64 dedicatedVideoMemory = desc.DedicatedVideoMemory;

					if (forceGPU == EForceGPU::AMD && (wcsstr(desc.Description, L"AMD") || wcsstr(desc.Description, L"Radeon")))
					{
						maxDedicatedVideoMemory = dedicatedVideoMemory;
						iAdapter = i;
						adapterDesc = desc;
						break;
					}

					if (dedicatedVideoMemory > maxDedicatedVideoMemory && (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0)
					{
						maxDedicatedVideoMemory = dedicatedVideoMemory;
						iAdapter = i;
						adapterDesc = desc;
					}
				}

				DXGIFactory->EnumAdapters1(iAdapter, &dxgiAdapter1);
			}
		}
		else
		{
			switch (forceGPU)
			{
				case EForceGPU::NVIDIA:
					printf("Forcing GPU: NVIDIA\n");
					break;
				case EForceGPU::AMD:
					printf("Forcing GPU: AMD\n");
					break;
				case EForceGPU::Intel:
					printf("Forcing GPU: Intel\n");
					break;
				default:
					break;
			}

			bool bFound = false;

			// an IHV is forced: find the first adapter from them
			for (u32 i = 0; DXGIFactory->EnumAdapters1(i, &dxgiAdapter1) != DXGI_ERROR_NOT_FOUND; i++)
			{
				DXGI_ADAPTER_DESC1 desc;
				dxgiAdapter1->GetDesc1(&desc);
				const u64 dedicatedVideoMemory = desc.DedicatedVideoMemory;

				if ((forceGPU == EForceGPU::NVIDIA && (wcsstr(desc.Description, L"NVIDIA") || wcsstr(desc.Description, L"GeForce"))) ||
					(forceGPU == EForceGPU::AMD && (wcsstr(desc.Description, L"AMD") || wcsstr(desc.Description, L"Radeon"))) ||
					(forceGPU == EForceGPU::Intel && wcsstr(desc.Description, L"Intel")))
				{
					bFound = true;
					maxDedicatedVideoMemory = dedicatedVideoMemory;
					iAdapter = i;
					adapterDesc = desc;
					break;
				}
			}

			assert(bFound);

			DXGIFactory->EnumAdapters1(iAdapter, &dxgiAdapter1);
		}

		static const D3D_FEATURE_LEVEL requiredFeatureLevels[] =
		{
			D3D_FEATURE_LEVEL_12_0,
			D3D_FEATURE_LEVEL_11_1,
			D3D_FEATURE_LEVEL_11_0
		};
		static const u32 nFeatureLevels = _countof(requiredFeatureLevels);

		D3D12_FEATURE_DATA_FEATURE_LEVELS featureLevels{};
		featureLevels.pFeatureLevelsRequested = requiredFeatureLevels;
		featureLevels.NumFeatureLevels = nFeatureLevels;

		for (u32 i = 0; i < nFeatureLevels; i++)
			if (SUCCEEDED(D3D12CreateDevice(dxgiAdapter1.Get(), requiredFeatureLevels[i], __uuidof(ID3D12Device), nullptr)))
			{
				OutMaxFeatureLevel = requiredFeatureLevels[i];
				break;
			}
			else
				assert(false);

		assert(dxgiAdapter1);
		assert(SUCCEEDED(dxgiAdapter1.As(&dxgiAdapter4)));

		printf("Using GPU: %ws\n", adapterDesc.Description);
	}

	return dxgiAdapter4;
}

i32 main(int argc, char* argv[])
{
	//wstring pixCapturerDLL = GetLatestWinPixGpuCapturerPath_Cpp17();
	//LoadLibrary(pixCapturerDLL.c_str());
	//PIXCaptureParameters pixParams{};
	//pixParams.GpuCaptureParameters.FileName = L"c:\\temp\\capture.wpix";
	//const HRESULT res = PIXBeginCapture(PIX_CAPTURE_GPU, &pixParams);
	//assert(SUCCEEDED(res));

	ComPtr<ID3D12Device> device;
	ComPtr<IDXGIFactory4> factory;
	D3D_FEATURE_LEVEL maxFL;
	ComPtr<IDXGIAdapter4> adapter = GetAdapter(maxFL, "intel", false);
	CreateDXGIFactory1(IID_PPV_ARGS(&factory));

	//EnableDebugLayer();

	assert(SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device))));

	//ComPtr<ID3D12InfoQueue> infoQueue;
	//if (SUCCEEDED(device.As(&infoQueue)))
	//{
	//	infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
	//	infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
	//	infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, TRUE);

	//	D3D12_MESSAGE_SEVERITY excludedSeverities[]{ D3D12_MESSAGE_SEVERITY_INFO };

	//	D3D12_INFO_QUEUE_FILTER newFilter{};
	//	newFilter.DenyList.NumSeverities = _countof(excludedSeverities);
	//	newFilter.DenyList.pSeverityList = excludedSeverities;

	//	assert(SUCCEEDED(infoQueue->PushStorageFilter(&newFilter)));
	//}

	ComPtr<ID3D12DescriptorHeap> svrDescriptorHeap;
	D3D12_DESCRIPTOR_HEAP_DESC svrDescHeapDesc{};
	svrDescHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	svrDescHeapDesc.NumDescriptors = 2048;
	svrDescHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	svrDescHeapDesc.NodeMask = 0;
	assert(SUCCEEDED(device->CreateDescriptorHeap(&svrDescHeapDesc, IID_PPV_ARGS(&svrDescriptorHeap))));

	// Create command queue
	D3D12_COMMAND_QUEUE_DESC queueDesc = {};
	queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
	ComPtr<ID3D12CommandQueue> commandQueue;
	device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue));

	ComPtr<ID3D12CommandAllocator> commandAllocator;
	device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocator));

	// Create command list
	ComPtr<ID3D12GraphicsCommandList> commandList;
	assert(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocator.Get(), nullptr, IID_PPV_ARGS(&commandList))));

	// Reset command list and allocator
	assert(SUCCEEDED(commandList->Close()));
	commandAllocator->Reset();
	commandList->Reset(commandAllocator.Get(), nullptr);

	// Create buffers
	std::vector<ComPtr<ID3D12Resource>> buffers;
	std::vector<ComPtr<ID3D12Resource>> uploadBuffers;
	CreateBuffers(device.Get(), buffers, uploadBuffers, svrDescriptorHeap.Get(), commandList.Get());

	// Load the compiled shader binary
	std::vector<char> shaderBinary = LoadShaderBinary(L"Test.cso");

	// Create root signature
	D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc = {};
	rootSignatureDesc.NumParameters = 1;
	D3D12_ROOT_PARAMETER rootParameters[1];
	rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
	rootParameters[0].Constants.ShaderRegister = 0;
	rootParameters[0].Constants.RegisterSpace = 0;
	rootParameters[0].Constants.Num32BitValues = 1;
	rootParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

	rootSignatureDesc.pParameters = rootParameters;
	rootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED;

	ComPtr<ID3DBlob> rootSignatureBlob;
	D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rootSignatureBlob, nullptr);

	ComPtr<ID3D12RootSignature> rootSignature;
	device->CreateRootSignature(0, rootSignatureBlob->GetBufferPointer(), rootSignatureBlob->GetBufferSize(), IID_PPV_ARGS(&rootSignature));

	// Create compute pipeline state object using the binary shader
	D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc = {};
	computePsoDesc.pRootSignature = rootSignature.Get();
	computePsoDesc.CS.pShaderBytecode = shaderBinary.data();
	computePsoDesc.CS.BytecodeLength = shaderBinary.size();

	ComPtr<ID3D12PipelineState> computePipelineState;
	device->CreateComputePipelineState(&computePsoDesc, IID_PPV_ARGS(&computePipelineState));

	// Execute compute shader
	ExecuteComputeShader(device.Get(), commandQueue.Get(), commandList.Get(), buffers, computePipelineState.Get(), rootSignature.Get(), svrDescriptorHeap.Get());

	DownloadAndCheckBuffers(device.Get(), commandQueue.Get(), buffers);

	//PIXEndCapture(FALSE);

	return 0;
}
