#include "cute/tensor.hpp"
#include "cute/util/print_latex.hpp"


int main() {
    using mma_op = cute::SM80_16x8x16_F32BF16BF16F32_TN;    // T stands for transpose, N stands for normal(col-major)
    using mma_traits = cute::MMA_Traits<mma_op>;
    using mma_atom = cute::MMA_Atom<mma_traits>;

    using MMA = decltype(cute::make_tiled_mma(mma_atom{}));
    
    // to check the compiled latex can use this website: https://www.overleaf.com
    cute::print_latex(MMA{});

    return 0;
}
