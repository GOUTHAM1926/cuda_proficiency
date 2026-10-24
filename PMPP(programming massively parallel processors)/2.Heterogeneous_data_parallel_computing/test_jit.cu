__global__ void my_kernel() {
    int i = threadIdx.x;
}
int main() {
    my_kernel<<<1, 1>>>();
    return 0;
}
