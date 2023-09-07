
template <class C> 
void ParFFT2D::trasform(std::vector<std::vector<C>>& input_matrix, std::vector<std::vector<std::complex<double>>>& freq_matrix, const unsigned int n_threads){

    // Check if the input matrix is empty and return if it is.
    if (input_matrix.empty())
        return;

    // Get the number of columns and rows in the input matrix.
    const unsigned int n_cols = input_matrix[0].size(), n_rows = input_matrix.size();

    // Calculate the number of bits needed for the FFT algorithm.
    unsigned int numBits = static_cast<unsigned int>(log2(n_cols));

    // Create temporary vectors for column-wise FFT processing.
    std::vector<std::complex<double>> col(n_rows, 0.0);
    std::vector<std::vector<std::complex<double>>> input_cols(n_cols, col);
    freq_matrix.resize(n_rows);

    // Compute the indexes to swap only once and save them in a map of n elements
    std::vector<unsigned int> swapIdxsRows(n_cols), swapIdxsCols;

    // Calculate the bit-reversal permutations for rows
    const unsigned int halfNumBits = (numBits >> 1) + (numBits & 1U);
    for (unsigned int l = 0; l < n_cols; l++) {
        unsigned int ji = 0;
        // Bit reversal code from 2 directions -> <-
        for (unsigned int k = 0; k < halfNumBits; k++) {
            unsigned int shiftFactor = (numBits - 1 - (k << 1U));
            ji |= ((l & (1U << k)) << shiftFactor) | ((l & (1U << (numBits - 1 - k))) >> shiftFactor);
        }

        swapIdxsRows.push_back(ji);
    }

    std::vector<unsigned int>& ref_swapIdxsCols = n_cols != n_rows ? swapIdxsCols : swapIdxsRows;

    // If the number of columns is not equal to the number of rows, calculate bit-reversal permutations for columns
    if (n_cols != n_rows) {
        for (unsigned int l = 0; l < n_rows; l++) {
            unsigned int ji = 0;
            for (unsigned int k = 0; k < log2(n_rows); k++) {
                ji = (ji << 1) | ((l >> k) & 1U);
            }
            if (ji > l) {
                swapIdxsCols.push_back(ji);
            }
        }
    }

    
    // First pass: Apply FFT to each row in parallel
    #pragma omp parallel for num_threads(n_threads) firstprivate(n_cols, n_rows, numBits)
    for (unsigned int i = 0; i < n_rows; ++i) {
        std::vector<std::complex<double>> &row_vector = freq_matrix[i];
        row_vector.resize(n_cols);

        // Apply the row-wise bit-reversal permutation to the input_matrix
        for (unsigned int j = 0; j < n_cols; ++j)
            std::swap(input_matrix[i][j],input_matrix[i][swapIdxsRows[j]]);

        // Use the last iteration to compute and store column vectors and the first iteration to avoid overwriting the original input_matrix
        // Perform FFT for the first stage (s = 1)
        {
            unsigned int m = 1U << 1; 
            std::complex<double> wm = std::exp(-2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
            for (unsigned int k = 0; k < n_cols; k += m) {

                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * static_cast<std::complex<double>>(input_matrix[i][k + j + m / 2]);
                    std::complex<double> u = static_cast<std::complex<double>>(input_matrix[i][k + j]);
                    row_vector[k + j] = u + t;
                    row_vector[k + j + m / 2] = u - t;
                    w *= wm;
                }
            }
        }
        // swap again to restore original input_matrix
        for (unsigned int j = 0; j < n_cols; ++j)
            std::swap(input_matrix[i][j],input_matrix[i][swapIdxsRows[j]]);


        for (unsigned int s = 2; s < numBits; s++) {
            unsigned int m = 1U << s; 
            std::complex<double> wm = std::exp(-2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
            for (unsigned int k = 0; k < n_cols; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * row_vector[k + j + m / 2];
                    std::complex<double> u = row_vector[k + j];
                    row_vector[k + j] = u + t;
                    row_vector[k + j + m / 2] = u - t;
                    w *= wm;
                }
            }
        }

        // Perform FFT for the last stage (s == numBits)
        {
            unsigned int m = 1U << numBits; 
            std::complex<double> wm = std::exp(-2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
            for (unsigned int k = 0; k < n_cols; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * row_vector[k + j + m / 2];
                    std::complex<double> u = row_vector[k + j];
                    input_cols[k + j][i] = u + t;
                    input_cols[k + j + m / 2][i] = u - t;
                    w *= wm;
                }
            }
        }
    }

    //Second pass: Apply FFT to each column
    numBits = static_cast<unsigned int>(log2(n_rows));

    #pragma omp parallel for num_threads(n_threads) firstprivate(n_cols, n_rows, numBits)
    for (unsigned int i = 0; i < n_cols; ++i) {
        std::vector<std::complex<double>> &col_vector = input_cols[i];

        for (unsigned int j = 0; j < n_rows; ++j)
            std::swap(col_vector[j],col_vector[ref_swapIdxsCols[j]]);
            

        for (unsigned int s = 1; s < numBits; s++) {
            unsigned int m = 1U << s; 
            std::complex<double> wm = std::exp(-2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
            for (unsigned int k = 0; k < n_rows; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * col_vector[k + j + m / 2];
                    std::complex<double> u = col_vector[k + j];
                    col_vector[k + j] = u + t;
                    col_vector[k + j + m / 2] = u - t;
                    w *= wm;
                }
            }
        }
        // s == numBits
        {
            unsigned int m = 1U << numBits; 
            std::complex<double> wm = std::exp(-2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
            for (unsigned int k = 0; k < n_rows; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * col_vector[k + j + m / 2];
                    std::complex<double> u = col_vector[k + j];
                    freq_matrix[k + j][i] = u + t;
                    freq_matrix[k + j + m / 2][i] = u - t;
                    w *= wm;
                }
            }
        }
    }
}

template <class C> 
void ParFFT2D::iTransform(std::vector<std::vector<std::complex<double>>>& input_matrix, std::vector<std::vector<C>>& space_matrix, const unsigned int n_threads){
    if (input_matrix.empty())
        return;
        
    const unsigned int n_cols = input_matrix[0].size(), n_rows = input_matrix.size();

    unsigned int numBits = static_cast<unsigned int>(log2(n_cols));

    std::vector<std::complex<double>> col(n_rows,0.0);
    std::vector<std::vector<std::complex<double>>> input_cols(n_cols,col);
    space_matrix.resize(n_rows);
    
    //First pass: Apply iFFT to each row
    #pragma omp parallel for num_threads(n_threads) firstprivate(n_cols, n_rows, numBits)
    for (unsigned int i = 0; i < n_rows; ++i) {
        space_matrix[i].resize(n_cols);
        std::vector<std::complex<double>> row_vector = input_matrix[i];
        
        for (unsigned int l = 0; l < n_cols; l++) { // **************
            unsigned int ji = 0;
            for (unsigned int k = 0; k < numBits; k++) {
                ji = (ji << 1) | ((l >> k) & 1U);
            }
            if (ji > l) {
                std::swap(row_vector[l], row_vector[ji]);
            }
        }

        for (unsigned int s = 1; s < numBits; s++) {
            unsigned int m = 1U << s; 
            std::complex<double> wm = std::exp(2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
            for (unsigned int k = 0; k < n_cols; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * row_vector[k + j + m / 2];
                    std::complex<double> u = row_vector[k + j];
                    row_vector[k + j] = u + t;
                    row_vector[k + j + m / 2] = u - t;
                    w *= wm;
                }
            }
        }
        // s == numBits
        {
            unsigned int m = 1U << numBits; 
            std::complex<double> wm = std::exp(2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
            for (unsigned int k = 0; k < n_cols; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * row_vector[k + j + m / 2];
                    std::complex<double> u = row_vector[k + j];
                    input_cols[k + j][i] = (u + t) / static_cast<double>(n_cols);
                    input_cols[k + j + m / 2][i] = (u - t) /  static_cast<double>(n_cols);
                    w *= wm;
                }
            }
        }

    }

    
    //Second pass: Apply FFT to each column
    numBits = static_cast<unsigned int>(log2(n_rows));
    #pragma omp parallel for num_threads(n_threads) firstprivate(n_cols, n_rows, numBits)
    for (unsigned int i = 0; i < n_cols; ++i) {
        std::vector<std::complex<double>> &col_vector = input_cols[i];
        
        for (unsigned int l = 0; l < n_rows; l++){
            unsigned int j = 0;
            for (unsigned int k = 0; k < numBits; k++) {
                j = (j << 1) | ((l >> k) & 1U);
            }
            if (j > l) {
                std::swap(col_vector[l], col_vector[j]);
            }
        }

        for (unsigned int s = 1; s < numBits; s++) {
            unsigned int m = 1U << s; 
            std::complex<double> wm = std::exp(2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
            for (unsigned int k = 0; k < n_rows; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * col_vector[k + j + m / 2];
                    std::complex<double> u = col_vector[k + j];
                    col_vector[k + j] = u + t;
                    col_vector[k + j + m / 2] = u - t;
                    w *= wm;
                }
            }
        }
        
        // s == numBits
        {
            unsigned int m = 1U << numBits; 
            std::complex<double> wm = std::exp(2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
            for (unsigned int k = 0; k < n_rows; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * col_vector[k + j + m / 2];
                    std::complex<double> u = col_vector[k + j];
                    
                    space_matrix[k + j][i] = static_cast<C>( (u + t).real() / static_cast<double>(n_rows) + 0.5);
                    space_matrix[k + j + m / 2][i] = static_cast<C>((u - t).real() / static_cast<double>(n_rows) + 0.5);
                    w *= wm;
                }
            }
        }
    }
}