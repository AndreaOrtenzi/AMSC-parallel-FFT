
template <class C>
void ParFFT2D::transform(std::vector<std::vector<C>>& input_matrix, std::vector<std::vector<std::complex<double>>>& freq_matrix, const unsigned int n_threads) {

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

    // First pass: Apply FFT to each row in parallel.
    #pragma omp parallel for num_threads(n_threads) firstprivate(n_cols, n_rows, numBits)
    for (unsigned int i = 0; i < n_rows; ++i) {
        std::vector<std::complex<double>> &row_vector = freq_matrix[i];
        row_vector.resize(n_cols);

        // Bit-reversal permutation for the input_matrix.
        for (unsigned int l = 0; l < n_cols; l++) {
            unsigned int ji = 0;
            for (unsigned int k = 0; k < numBits; k++) {
                ji = (ji << 1) | ((l >> k) & 1U);
            }
            if (ji > l) {
                std::swap(input_matrix[i][l], input_matrix[i][ji]);
            }
        }

        // Perform the FFT for each row.
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

        // Bit-reversal permutation to restore the original input_matrix order.
        for (unsigned int l = 0; l < n_cols; l++) {
            unsigned int ji = 0;
            for (unsigned int k = 0; k < numBits; k++) {
                ji = (ji << 1) | ((l >> k) & 1U);
            }
            if (ji > l) {
                std::swap(input_matrix[i][l], input_matrix[i][ji]);
            }
        }

        // Perform FFT for other stages.
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

        // Special case for the final stage to directly fill input_cols (s == numBits).
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

    // Second pass: Apply FFT to each column.
    numBits = static_cast<unsigned int>(log2(n_rows));

    #pragma omp parallel for num_threads(n_threads) firstprivate(n_cols, n_rows, numBits)
    for (unsigned int i = 0; i < n_cols; ++i) {
        std::vector<std::complex<double>> &col_vector = input_cols[i];

        // Bit-reversal permutation for the input_cols.
        for (unsigned int l = 0; l < n_rows; l++) {
            unsigned int j = 0;
            for (unsigned int k = 0; k < numBits; k++) {
                j = (j << 1) | ((l >> k) & 1U);
            }
            if (j > l) {
                std::swap(col_vector[l], col_vector[j]);
            }
        }

        // Perform the FFT for each column.
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

        // Special case for the final stage to directly fill freq_matrix (s == numBits).
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
void ParFFT2D::iTransform(std::vector<std::vector<std::complex<double>>>& input_matrix, std::vector<std::vector<C>>& space_matrix, const unsigned int n_threads) {

    if (input_matrix.empty())
        return;

    // Get the number of columns and rows in the input matrix.
    const unsigned int n_cols = input_matrix[0].size(), n_rows = input_matrix.size();

    // Calculate the number of bits needed for the FFT algorithm.
    unsigned int numBits = static_cast<unsigned int>(log2(n_cols));

    // Create temporary vectors for column-wise iFFT processing.
    std::vector<std::complex<double>> col(n_rows, 0.0);
    std::vector<std::vector<std::complex<double>>> input_cols(n_cols, col);
    space_matrix.resize(n_rows);

    // First pass: Apply iFFT to each row in parallel.
    #pragma omp parallel for num_threads(n_threads) firstprivate(n_cols, n_rows, numBits)
    for (unsigned int i = 0; i < n_rows; ++i) {
        space_matrix[i].resize(n_cols);
        std::vector<std::complex<double>> row_vector = input_matrix[i];

        // Bit-reversal permutation for the input_matrix.
        for (unsigned int l = 0; l < n_cols; l++) {
            unsigned int ji = 0;
            for (unsigned int k = 0; k < numBits; k++) {
                ji = (ji << 1) | ((l >> k) & 1U);
            }
            if (ji > l) {
                std::swap(row_vector[l], row_vector[ji]);
            }
        }

        // Perform the iFFT for each row.
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

        // Special case for the final stage to directly fill input_cols (s == numBits).
        {
            unsigned int m = 1U << numBits;
            std::complex<double> wm = std::exp(2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
            for (unsigned int k = 0; k < n_cols; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * row_vector[k + j + m / 2];
                    std::complex<double> u = row_vector[k + j];
                    input_cols[k + j][i] = (u + t) / static_cast<double>(n_cols);
                    input_cols[k + j + m / 2][i] = (u - t) / static_cast<double>(n_cols);
                    w *= wm;
                }
            }
        }
    }

    // Second pass: Apply FFT to each column.
    numBits = static_cast<unsigned int>(log2(n_rows));
    #pragma omp parallel for num_threads(n_threads) firstprivate(n_cols, n_rows, numBits)
    for (unsigned int i = 0; i < n_cols; ++i) {
        std::vector<std::complex<double>> &col_vector = input_cols[i];

        // Bit-reversal permutation for the input_cols.
        for (unsigned int l = 0; l < n_rows; l++) {
            unsigned int j = 0;
            for (unsigned int k = 0; k < numBits; k++) {
                j = (j << 1) | ((l >> k) & 1U);
            }
            if (j > l) {
                std::swap(col_vector[l], col_vector[j]);
            }
        }

        // Perform the FFT for each column.
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

        // Special case for the final stage to directly fill space_matrix (s == numBits).
        {
            unsigned int m = 1U << numBits;
            std::complex<double> wm = std::exp(2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
            for (unsigned int k = 0; k < n_rows; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * col_vector[k + j + m / 2];
                    std::complex<double> u = col_vector[k + j];
                    space_matrix[k + j][i] = static_cast<C>((u + t).real() / static_cast<double>(n_rows) + 0.5);
                    space_matrix[k + j + m / 2][i] = static_cast<C>((u - t).real() / static_cast<double>(n_rows) + 0.5);
                    w *= wm;
                }
            }
        }
    }
}
