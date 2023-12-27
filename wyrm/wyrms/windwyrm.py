        if self.wlen < 0:
            raise ValueError('window_sec must be positive')

        if self.slen < 0:
            raise ValueError('stride_sec must be positive')



        if isinstance(window_sec, float):
            self.wlen = window_sec
        elif isinstance(window_sec, int):
            self.wlen = float(window_sec)
        else:
            raise TypeError('window_sec must be type float or int')

        if isinstance(stride_sec, float):
            self.slen = stride_sec
        elif isinstance(stride_sec, int):
            self.slen = float(stride_sec)
        else:
            raise TypeError('stride_sec must be type float or int')