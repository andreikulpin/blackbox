all dep clean tests::
	cd brute && $(MAKE) $@ && cd .. 

doc: indent doxy

doxy:
	mkdir -p doc/html &&\
	doxygen doxy.conf

clean::
	rm -rf *~ PI* core bin/* obj/* tmp *.log