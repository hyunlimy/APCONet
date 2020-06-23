import math
import numpy as np
from numbers import Number
import copy


def print_all(data):
    """
    :param data:
    :return: print full array
    """
    print('[' + ','.join([str(x) for x in data]) + ']')


def corr(a, b):
    """
    :param a:
    :param b:
    :return: correlation coefficient between two data
    """
    return np.corrcoef(a, b)[0, 1]


def max_idx(data, idxfrom=0, idxto=None):
    """
    :param data:
    :param idxfrom:
    :param idxto:
    :return:
    """

    idxfrom = max(0, idxfrom)
    if idxto == None:
        idxto = len(data)
    idxto = min(len(data), idxto)
    return idxfrom + np.argmax(data[idxfrom:idxto])


def min_idx(data, idxfrom=0, idxto=None):
    """
    :param data:
    :param idxfrom:
    :param idxto:
    :return:
    """
    idxfrom = max(0, idxfrom)
    if idxto is None:
        idxto = len(data)
    idxto = min(len(data), idxto)
    return idxfrom + np.argmin(data[idxfrom:idxto])


def get_samples(data, srate, idxes):
    """
    Gets a sample of the wave with the indexes specified by idxes
    returns has a form of [{'dt' : , 'val' : }, ... ]
    :param data:
    :param srate:
    :param idxes:
    :return:
    """
    return [{"dt": idx / srate, "val": data[idx]} for idx in idxes]


def is_num(x):
    """
   
    :param x:
    :return:
    """
    if not isinstance(x, Number):
       return False
    return math.isfinite()


def exclude_undefined(data):
    """

    :param data:
    :return:
    """
    ret = []
    for x in data:
        if is_num(x):
            ret.append(x)
    return ret


def extend_undefined(data):
    """

    :param data:
    :return:
    """

    # Remove undefined left
    for i in range(len(data)):
        if is_num(data[i]): # find the value which is not undefined
            # Change all previous values to the value
            data = [data[i]] * i + data[i:]
            break
        if i == len(data)-1: # Set all to 0 if all are undefined
            return [0] * len(data)

    # Remove undefined right
    for i in range(len(data)-1, -1, -1):
        if is_num(data[i]):
            data = data[:i] + [data[i]] * (len(data)-1)
            break
    return


def interp_undefined(data):
    """

    :param data:
    :return:
    """
    data = extend_undefined(data)
    if len(data) <=2:
        return data
    lastpos = 0
    lastval = data[0]
    for i in range(1, len(data)):
        if is_num(data[i]):
            for j in range(lastpos+1, i):
                data[j] = (data[i] - lastval) * (j-lastpos) / (i-lastpos) + lastval  # fill values between them
            lastpos = i
            lastval = data[i]
    return data


def replace_undefined(data):
    """

    :param data:
    :return:
    """
    data = extend_undefined(data)
    if len(data)<=2:
        return data
    lastpos = 0
    lastval = data[0]
    for i in range(1, len(data)):
        if is_num(data[i]):
            for j in range(lastpos + 1, i):
                data[j] = lastval
            lastpos = i
            lastval = data[i]
    return data


def detect_window_maxima(data, wind):
    """
    Returns the index of the sample which is the maximum from the samples in the specified window.
    :param data:
    :param wind:
    :return:
    """
    span = int(wind / 2)
    ret = []
    i = span
    while i < len(data) - span: # Because i cannot be changed within a for statement
        idx_max = max_idx(data, i - span, i + span)
        if idx_max == i:
            ret.append(i)
        elif i < idx_max:
            i = idx_max -1 #jump
        i += 1
    return ret


def detect_window_minimum(data, wind):
    """
    Returns the index of the sample which is the minimum from the samples in the specified window.
    :param data:
    :param wind:
    :return:
    """
    span = int(wind /2)
    ret = []
    i = span
    while i < len(data) - span: # Because i cannot be changed within a for statement
        ifrom = max(0, i-span)
        ito = min(i + span, len(data))
        idx_min = ifrom + np.argmin(data[ifrom:ito])
        if idx_min == i:
            ret.append(i)
        elif i < idx_min:
            i = idx_min -1 # jump
        i += 1
    return ret


def detect_maxima(data, tr = 0):
    """
    Find indexes of x such that xk=1 <= x >= xk +1
    :param data: arr
    :param tr: percentile threshold (0-100)
    :return: detect peak above tr
    """
    tval = np.percentile(data, tr)
    ret = []
    for i in range(1, len(data) -1):
        if data[i-1] < data[i]: # increased value compared to previous value
            if data[i] > tval:
                is_peak = False
                for j in range(i+1, len(data)): # find next increase of decrease
                    if data[i] == data[j]:
                        continue
                    if data[i] < data [j]:
                        break # value increased -> not a peak
                    if data[i] > data[j]:
                        is_peak = True
                        break # value decreased -> peak!
                if is_peak:
                    ret.append(i)
    return ret


def detect_minima(data, tr = 100):
    """
    Find indexed of x such that xk-1 <= x >= xk+1
    :param data: arr
    :param tr: percentile threshold (0-100)
    :return: detect peak above tr
    """
    tval = np.percentile(data, tr)
    ret = []
    for i in range(1, len(data)-1):
        if data[i-1] > data[i]: # value decreased
            if data[i] < tval:
                is_nadir = False
                for j in range(i+1, len(data)): # find next increased of decreased
                    if data[i] == data[j]:
                        continue
                    if data[i] > data[j]:
                        break # value increased -> minima!
                    if data[i] < data[j]:
                        is_nadir = True
                        break
                if is_nadir:
                    ret.append(i)

    return ret


def next_power_of_2(x):
    """
    Find power of 2 greater than x
    :param x:
    :return:
    """
    return 2 ** math.ceil(math.log(x) / math.log(2))


def band_pass(data, srate, fl, fh):
    """
    band pass filter
    :param data:
    :param srate:
    :param fl:
    :param fh:
    :return:
    """
    if fl > fh:
        return band_pass(data, srate, fh, fl)

    oldlen = len(data)
    newlen = next_power_of_2(oldlen)

    # srate / sxamp = Frequency increment
    # (0~ nsamp-1) * srate / nsamp = frequency range
    y = np.fft.fft(data, newlen)

    # filtering
    half = math.ceil(newlen / 2)
    for i in range(half):
        f = i * srate / newlen
        if f < fl or f > fh:
            y[i] = y[newlen - 1 - i] = 0

    # inverse transform
    return np.real(np.fft.ifft(y)[:oldlen])


def find_nearest(data, value):
    """
    Find the nearest value and return it
    :param data: array
    :param value: value to find
    :return: nearest value
    """
    idx = np.abs(np.array(data) - value).argmin()
    return data[idx]


def detect_qrs(data, srate):
    """
    Find qrs and return the indexes
    http://ocw.utm.my/file.php/38/SEB4223/07_ECG_Analysis_1_-_QRS_Detection.ppt%20%5BCompatibility%20Mode%5D.pdf
    :param data:
    :param srate:
    :return:
    """
    pcand = detect_window_maxima(data, 0.08*srate) # 80ms local peak
    y1 = band_pass(data, srate, 10, 20) # The qrs valud must be one of these
    y2 = [0, 0]
    for i in range(2, len(y1)-2):
        y2.append(2 * y1[i+2] + y1[i+1] - y1[i-1] - 2* y1[i-2])
    y2 += [0 , 0]
    y3 = [(x * x) for x in y2] # Squaring

    # Moving Average filter
    # Acts as a smoother and performs a moving window integrator over 150 ms
    y4 = []
    size = srate * 0.15
    for i in range(len(y3)):
        ifrom = max(0, i - int(size / 2))
        ito = min(i + int(size / 2), len(y3))
        y4.append(np.mean(y3[ifrom:ito]))

    y5 = band_pass(y4, srate, 0.5, 8) # filtering -> remove drifting and abrupt change
    p1 = detect_window_maxima(y5, 0.3 * srate) # find max in 300ms

    p2 = []
    for idx in p1:
        val = y5[idx]
        peak_vals = []
        for idx2 in p1:
            if abs(idx - idx2) < srate * 10:
                peak_vals.append(y5[idx])
            th = np.median(peak_vals) * 0.75
            if val >= th:
                p2.append(idx)

    # Find closest peak
    p3 = []
    last = -1
    for x in p2:
        idx_cand = find_nearest(pcand, x)
        if idx_cand != last:
            p3.append(idx_cand)
        last = idx_cand

    #remove false positives (FP)
    i = 0
    while i < len(p3) - 1:
        idx1 = p3[i]
        idx2 = p3[i+1]
        if idx2 - idx1 < 0.2 * srate:
            if i == 0:
                dele = i
            elif i >= len(p3) - 2:
                dele = i + 1
            else: #minimize heart rate variability
                idx_prev = p3[i - 1]
                idx_next = p3[i + 2]
                #find center point distance
                if abs(idx_next + idx_prev - 2 * idx1) > abs(idx_next + idx_prev - 2 * idx2):
                    dele = i
                else:
                    dele = i + 1
            p3.pop(dele)
            if dele == i:
                i -= 1
        i += 1

    return p3


def remove_wander_spline(data, srate):
    """
    cubic spline ECG wander removal
    http://jh786t2saqs.tistory.com/416
    http://blog.daum.net/jty71/10850833
    :param data:
    :param srate:
    :return:
    """
    # calculate downslope
    downslope = [0, 0, 0]
    for i in range(3, len(data) - 3):
        downslope.append(data[i-3] + data[i-1] - data[i+1] - data[i+3])
    downslope += [0, 0, 0]

    r_list = detect_qrs(data, srate)  # detect r-peak

    rsize = int(0.060 * srate)  # knots from r-peak
    jsize = int(0.066 * srate)
    knots = []  # indexes of the kot
    for ridx in r_list:
        th = 0.6 * max(downslope[ridx:ridx + rsize])
        for j in range(ridx, ridx + rsize):
            if downslope[j] >= th:  # R detected
                knots.append(j - jsize)
                break

    baseline = [0] * len(data)
    for i in range(1, len(knots) - 2):
        # x1, x2 = knots[i:i+1]
        x1 = knots[i]
        x2 = knots[i+1]
        # y1, y2 = data[x1:x2]
        y1 = data[x1]
        y2 = data[x2]
        d1 = (data[x2] - data[knots[i-1]]) / (x2 - knots[i-1])
        d2 = (data[knots[i+2]] - data[x1]) / (knots[i+2] - x1)
        a = -2 * (y2-y1) / (x2-x1)**3 + (d2+d1) / (x2-x1)**2
        b = 3 * (y2-y1) / (x2-x1)**2 - (d2+2*d1) / (x2 - x1)
        c = d1
        d = y1
        for x in range(x1, x2):
            x_a = (x - x1)  # x-a
            x_a2 = x_a * x_a
            x_a3 = x_a2 * x_a
            baseline[x] = a * x_a3 + b * x_a2 + c * x_a + d

    for i in range(len(data)):
        data[i] -= baseline[i]

    return data


def resample(data, dest_len, avg = False):
    """
    Remove the data
    :param data:
    :param dest_len:
    :param avg: if Ture, the data is averaged and resampled (slower)
    applied only for downsamplin. It is meaningless for upsampling
    :return:
    """
    if dest_len == 0:
        return []

    src_len = len(data)
    if src_len == 0:
        return [0] * dest_len

    if dest_len == 1:  # average
        if avg:
            return [np.mean(data)]
        else:
            return [data[0]]

    if src_len == 1:  # copy
        return [data[0]]

    if src_len == dest_len:
        return copy.deepcopy(data)

    if src_len < dest_len:  # upsample -> linear interpolate
        ret = []
        for x in range(dest_len):
            srcx = x / (dest_len - 1) * (src_len - 1)  # current position of x
            srcx1 = math.floor(srcx)  # index 1 on the original
            srcx2 = math.ceil(srcx)  # index 2 on the original
            factor = srcx - srcx1  # how close to index 2
            val1 = data[srcx1]
            val2 = data[srcx2]
            ret.append(val1 * (1 - factor) + val2 * factor)

        return ret

    # if src_len > dest_len: # downsample -> nearest or avg
    if avg:
        ret = []
        for x in range(dest_len):
            src_from = int(x * src_len / dest_len)
            src_to = int((x + 1) * src_len / dest_len)
            ret.append(np.mean(data[src_from:src_to]))
        return ret

    ret = []
    for x in range(dest_len):
        srcx = int(x * src_len / dest_len)
        ret.append(data[srcx])
    return ret


def resample_hz(data, srate_from, srate_to, avg = False):
    """

    :param data:
    :param srate_from:
    :param srate_to:
    :param avg:
    :return:
    """
    dest_len = int(math.ceil(len(data) / srate_from * srate_to))
    return resample(data, dest_len, avg)


def estimate_heart_freq(data, srate, fl = 30/60, fh = 200/60):
    """
    An automatic beat detection algorithm for pressure signal
    http://www.ncbi.nlm.nih.gov/pubmed/16235652
    :param data: input signal
    :param srate: sampling rate
    :param fl: lower bound of freq
    :param fh: upper bound of freq
    :return:
    """
    # Fourier transformed, and squared to obtain a frequency-dependent power
    # estimate psd in data
    p = abs(np.fft.fft(data)) ** 2
    maxf = 0
    maxval = 0
    for w in range(len(data)):
        f = w * srate / len(data)
        # add 11 harmonics, which do not exceed double of the default power
        # sampling
        if fl <= f <= fh:
            h = 0  # harmonic pds
            for k in range(1, 12):
                h += min(2 * p[w], p[(k * w) % len(data)])
            if h > maxval:
                maxval = h
                maxf = f
    return maxf

def detect_peaks(data, srate, mode = False):
    """
    obtain maximum and minimum values from blood pressure or pleth waveform
    the first min may not be found because all max is found and min is found in front of it
    for the reason, the first max dose not look for the min before it
    and the minlist is always one less than the maxlist
    :param data:
    :param srate:
    :return:
    """
    ret = []

    raw_data = copy.deepcopy(data)
    raw_srate = srate

    # resampling rate to 100 Hz
    data = resample_hz(data, srate, 100)
    srate = 100

    # upper and lower bound of the heart rate (Hz = /sec)
    # heart rate = hf * 60
    fh = 200 / 60  # 3.3
    fl = 30 / 60  # 0.5

    # estimate hr
    y1 = band_pass(data, srate, 0.5 * fl, 3 * fh)

    # Whole heart freq estimation
    if mode:
        hf = 75 / 60
    else:
        hf = estimate_heart_freq(y1, srate)
        if hf == 0:
            print("HR estimation failed, assume 75")
            hf = 75 / 60

    # band pass filter again with heart freq estimation
    y2 = band_pass(data, srate, 0.5 * fl, 2.5 * hf)
    d2 = np.diff(y2)

    # detect peak in gradient
    p2 = detect_maxima(d2, 90)

    # detect real peak
    y3 = band_pass(data, srate, 0.5 * fl, 10 * hf)
    p3 = detect_maxima(y3, 60)

    # find closest p3 that follows p2
    p4 = []
    last_p3 = 0
    for idx_p2 in p2:
        idx_p3 = 0
        for idx_p3 in p3:
            if idx_p3 > idx_p2:
                break
        if idx_p3 != 0:
            if last_p3 != idx_p3:
                p4.append(idx_p3)
                last_p3 = idx_p3

    # nearest neighbor and inter beat interbal (IBI) correction
    # p : location of detected peaks
    pc = []

    # find all maxima before preprocessing
    m = detect_maxima(data, 0)

    # correct peaks location error due to preprocessing
    last = -1
    for idx_p4 in p4:
        cand = find_nearest(m, idx_p4)
        if cand != last:
            pc.append(cand)
            last = cand

    ht = 1 / hf  # beat interval (sec)

    # correct false nagative (FN)
    # Make sure if there is rpeak not included in the PC
    i = -1
    while i < len(pc):
        idx_from = 0
        if i >= 0:
            idx_from = pc[i]
        idx_to = len(data)-1
        if i < len(pc)-1:
            idx_to = pc[i+1]

        # find false nagative and fill it
        if idx_to - idx_from < 1.75 * ht * srate:
            i += 1
            continue

        # It cannot be within 0.2 of both sides
        idx_from += 0.2 * ht * srate
        idx_to -= 0.2 * ht * srate

        # Find missing peak ans add it
        # find the maximum value from idx_from to idx_to
        idx_max = -1
        val_max = 0
        for idx_cand in m:
            if idx_cand <= idx_from:
                continue
            if idx_cand >= idx_to:
                break
            if idx_max == -1 or val_max < data[idx_cand]:
                val_max = data[idx_cand]
                idx_max = idx_cand

        # There is no candidate to this FN. Overtake
        if idx_max != -1:  # add idx_max and restart from there
            pc.insert(i+1, idx_max)
            i -= 1
        i += 1

    # correct false posirive (FP)
    i = 0
    while i < len(pc) -1:
        idx1 = pc[i]
        idx2 = pc[i+1]
        if idx2 - idx1 < 0.75 * ht * srate:  # false positive
            dele = i + 1  # default: delete i + 1
            if 1 < i < len(pc) -2:
                #minimize heart rate variability
                idx_prev = pc[i-1]
                idx_next = pc[i+2]

                # find center point distance
                d1 = abs(idx_next + idx_prev - 2 * idx1)
                d2 = abs(idx_next + idx_prev - 2 * idx2)

                if d1 > d2:
                    dele = i
                else:
                    dele = i+1

            elif i == 0:
                dele = i
            elif i == len(pc)-2:
                dele = i+1

            pc.pop(dele)
            i -= 1
        i += 1

    # remove duplicates
    i = 0
    for i in range(0, len(pc) -1):
        if pc[i] == pc[i+1]:
            pc.pop(i)
            i -= 1
        i += 1

    #find nearest peak in real data
    # We downsample x to srate to got maxidx, ex)1000 Hz -> 100 Hz.
    # Therefore, the position found by maxidx may differ by raw_srate / srate
    maxlist = []
    convpos = math.ceil(raw_srate / srate / 2)
    for maxidx in pc:
        idx = int(maxidx * raw_srate / srate)  # estimate idx -> not percise
        maxlist.append(max_idx(raw_data, idx - convpos - 1, idx + int(raw_srate / srate) + convpos + 1))

    # get the minlist from maxlist
    minlist = []
    for i in range(len(maxlist) - 1):
        minlist.append(min_idx(raw_data, maxlist[i], maxlist[i+1]))

    return [minlist, maxlist]


def estimate_resp_rate(data, srate):
    """
    count-adv algorithm
    doi: 10.1007/s10439-007-9428-1
    :param data:
    :param srate:
    :return:
    """
    filted = band_pass(data, srate, 0.1, 0.5)

    # find maxima
    maxlist = detect_maxima(filted)
    minlist = []  # find minima
    for i in range(len(maxlist) -1):
        minlist.append(min_idx(data, maxlist[i] + 1, maxlist[i+1]))
    extrema = maxlist + minlist
    extrema.sort()  # min, max, min, max

    while len(extrema) >= 4:
        diffs = []
        for i in range(len(extrema)-1):
            diffs.append(abs(filted[extrema[i]] - filted[extrema[i+1]]))
        th = 0.1 * np.percentile(diffs, 75)
        minidx = np.argmin(diffs)
        if diffs[minidx] >= th:
            break
        extrema.pop(minidx)
        extrema.pop(minidx)

    if len(extrema) < 3:
        print("warning: rr estimation failed, 13 used")
        return 13

    # Obtain both even-numbered or odd-numbered distances
    resp_len = np.mean(np.diff(extrema))* 2
    rr = 60 * srate / resp_len

    return rr


def detect_hr(data, srate, mod = 'max', rounds = 2, force_hf = False):
    """
    compute heart rate from arterial blood pressure with peak detection
    :param data: segments
    :param srate: hz of segments (data point per seconds)
    :param mod: 'max', 'min', 'mean'
    :return: hr (beats/minute)
    """

    try:
        maxpeaks, minpeaks = detect_peaks(data, srate, mode = force_hf)
    except ValueError:
        return round(0., rounds)

    maxpeaks = np.array(maxpeaks)
    minpeaks = np.array(minpeaks)

    if mod == 'min':
        # compute hr based on min peaks intervals
        min_intervals = np.diff(minpeaks)
        mean_beat_sec = np.median(min_intervals)
        beat_per_sec = srate / mean_beat_sec
        hr = round(beat_per_sec * 60, rounds)

    elif mod == 'mean':
        # compute hr based on average of max and min peaks intervals
        min_intervals = np.diff(minpeaks)
        max_intervals = np.diff(maxpeaks)
        mean_beat_sec = np.median(np.concatenate([min_intervals, max_intervals]))
        beat_per_sec = srate / mean_beat_sec
        hr = round(beat_per_sec * 60, rounds)

    else:
        if not mod == 'max':
            print('mod selection got wrong parameter: {}, Use "max" as default settings'.format(mod))
        # compute hr based on max peaks interval
        max_intervals = np.diff(maxpeaks)
        mean_beat_sec = np.median(max_intervals)
        beat_per_sec = srate / mean_beat_sec
        hr = round(beat_per_sec * 60, rounds)

    return hr


