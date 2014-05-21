#!/usr/bin/env python

"""
Retrieve operating parameters of connected USRP and loop through the operating spectrum trasmitting a constant wave signal
"""
from gnuradio import gr
from gnuradio import analog
from gnuradio import uhd
from time import sleep

MAX_RATE = 1000e6

class build_block(gr.top_block):
	def __init__(self):
		gr.top_block.__init__(self)

		args = "" #only supporting USB USRPs for now

		#find uhd devices

		d = uhd.find_devices(uhd.device_addr(args))
		if d:

			uhd_type = d[0].get('type')
			print "\nFound '%s'" % uhd_type
		else:
			print "\nNo device found"
			return

		#check version of USRP and set num_channels

		if uhd_type == "usrp":
			tx_nchan = 2
			rx_nchan = 2
		else:
			tx_nchan = 1
			rx_nchan = 1

		#setup transmit chain (usrp sink, signal source)
		
		#usrp sink
		stream_args = uhd.stream_args('fc32', channels = range(tx_nchan))
		self.u_tx = uhd.usrp_sink(device_addr=args, stream_args=stream_args)
		self.u_tx.set_samp_rate(MAX_RATE)

		#analog signal source - sig_source_c(sampling_freq,waveform, wave_freq, ampl, offset=0)
		self.tx_src0 = analog.sig_source_c(self.u_tx.get_samp_rate(), analog.GR_CONST_WAVE, 0, 1.0, 0)

		#check and output freq range, gain range, num_channels

		#gain range and max
		tx_gain_range = self.u_tx.get_gain_range()
		tx_gain_min = tx_gain_range.start()
		tx_gain_max = tx_gain_range.stop()

		#freq range
		tx_freq_range = self.u_tx.get_freq_range()
		tx_freq_low = tx_freq_range.start()
		tx_freq_high = tx_freq_range.stop()
		tx_freq_mid = (tx_freq_low + tx_freq_high) / 2.0

		#output info
		print "\nDevice Info"
		print "\n\tType: %s" % uhd_type

		print "\n\tMin Freq: %d MHz" % (tx_freq_low/1e6)
		print "\tMax Freq: %d MHz" % (tx_freq_high/1e6)
		print "\tMid Freq: %d MHz" % (tx_freq_mid/1e6)

		print "\n\tMin Gain: %d dB" % tx_gain_min
		print "\tMax Gain: %d dB" % tx_gain_max

		#set initial parameters 

		for i in xrange(tx_nchan):
			self.u_tx.set_center_freq(tx_freq_mid + i*1e6, i)
			self.u_tx.set_gain(tx_gain_max, i)

		#connect blocks

		self.connect(self.tx_src0, self.u_tx)

def main():
	try:
		tb = build_block()
		tb.start()
		print "Transmission test will cycle once through the operating frequencies hopping 10 MHz at a time"
		raw_input("Press Enter to begin transmission test & Ctrl-C to exit\n")

		start = tb.u_tx.get_freq_range().start()
		stop = tb.u_tx.get_freq_range().stop()

		freq_hops = int((stop - start) / 10e6) + 1	
		
		print "\nTransmit Frequencies:"

		channel = 0 #default to first channel

		for i in xrange(freq_hops):
			trans_freq = start + i * 10e6
			tb.u_tx.set_center_freq(trans_freq,channel)
			print "\n%d MHz" % (trans_freq/1e6)
			sleep(.3)
		
		print "\nTest Over"

		tb.stop()
	except [[KeyboardInterrupt]]:
		pass


if __name__ == '__main__':
	main()
