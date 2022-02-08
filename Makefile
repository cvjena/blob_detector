install:
	pip install . --no-deps --upgrade

new_env:
	conda create -n detector python~=3.9.0 opencv~=4.5.0 -y
	pip install --upgrade pip
	pip install -r requirements.txt

build_sdist:
	@python setup.py build sdist

deploy: build_sdist
	./deploy_latest.sh

test_deploy: build_sdist
	REPO=pypitest ./deploy_latest.sh

get_version:
	@printf v
	@python setup.py -V
