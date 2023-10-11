# Configuration

$PROJECT = $GITHUB_REPO  = 'safe'
$GITHUB_ORG = 'valence-labs'
$PUSH_TAG_REMOTE = 'git@github.com:valence-labs/safe.git'
$GHRELEASE_TARGET = 'main'

# Logic

$AUTHORS_FILENAME = 'AUTHORS.rst'
$AUTHORS_METADATA = '.authors.yml'
$AUTHORS_SORTBY = 'alpha'
$AUTHORS_MAILMAP = '.mailmap'

$CHANGELOG_FILENAME = 'CHANGELOG.rst'
$CHANGELOG_TEMPLATE = 'TEMPLATE.rst'
$CHANGELOG_NEWS = 'news'

$FORGE_FEEDSTOCK_ORG = 'valence-forge'
$FORGE_RERENDER = True
$FORGE_USE_GIT_URL = True
$FORGE_FORK = False
$FORGE_PULL_REQUEST = False

$ACTIVITIES = ['check', 'authors', 'version_bump', 'changelog', 'tag', 'push_tag', 'ghrelease', 'forge']

$VERSION_BUMP_PATTERNS = [('safe/_version.py', r'__version__\s*=.*', "__version__ = \"$VERSION\"")]