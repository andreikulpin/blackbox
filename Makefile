all dep clean tests::
	cd brute && $(MAKE) $@ && cd .. 
	cd rosenbrock && $(MAKE) $@ && cd ..
	cd methods/best_point && $(MAKE) $@ && cd ..

doc: indent doxy

doxy:
	mkdir -p doc/html &&\
	doxygen doxy.conf

clean::
	rm -rf *~ PI* core bin/* obj/* tmp *.log