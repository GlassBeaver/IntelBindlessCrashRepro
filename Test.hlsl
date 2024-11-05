cbuffer RootConsts : register(b0)
{
	uint Index;
}

[numthreads(1024, 1, 1)]
void main(uint DTid : SV_DispatchThreadID)
{
	RWStructuredBuffer<uint> data = ResourceDescriptorHeap[Index];
	data[DTid] += 1;
}
