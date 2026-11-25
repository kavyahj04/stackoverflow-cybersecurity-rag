import type { HistoryItem } from "../history";

interface Props {
  history: HistoryItem[];
  onPick: (item: HistoryItem) => void;
  onClear: () => void;
}

export default function Sidebar({ history, onPick, onClear }: Props) {
  return (
    <aside className="sidebar" aria-label="Recent questions">
      <div className="sidebar-title">Recent</div>
      {history.length === 0 ? (
        <p className="sidebar-empty">Your questions appear here. They are saved in this browser only.</p>
      ) : (
        <>
          <ul>
            {history.map((item) => (
              <li key={`${item.mode}:${item.q}`}>
                <button type="button" onClick={() => onPick(item)} title={item.q}>
                  {item.mode === "search" && <span className="sidebar-tag">search</span>}
                  {item.q}
                </button>
              </li>
            ))}
          </ul>
          <button type="button" className="link-button small" onClick={onClear}>
            Clear history
          </button>
        </>
      )}
    </aside>
  );
}
